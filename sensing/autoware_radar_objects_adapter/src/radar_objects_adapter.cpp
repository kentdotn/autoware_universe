// Copyright 2026 The Autoware Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "radar_objects_adapter.hpp"

#include <autoware_utils_geometry/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace autoware::radar_objects_adapter
{
namespace
{
using RadarClassification = autoware_sensing_msgs::msg::RadarClassification;
using ObjectClassification = autoware_perception_msgs::msg::ObjectClassification;
using DETECTION_COV_IDX = autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
using RADAR_COV_IDX = autoware_utils_geometry::xyz_upper_covariance_index::XYZ_UPPER_COV_IDX;

// Maps for classification remapping
const std::map<std::string, std::uint8_t> RADAR_LABEL_TO_UINT_MAP = {
  {"UNKNOWN", RadarClassification::UNKNOWN}, {"CAR", RadarClassification::CAR},
  {"TRUCK", RadarClassification::TRUCK},     {"MOTORCYCLE", RadarClassification::MOTORCYCLE},
  {"BICYCLE", RadarClassification::BICYCLE}, {"PEDESTRIAN", RadarClassification::PEDESTRIAN},
  {"ANIMAL", RadarClassification::ANIMAL},   {"HAZARD", RadarClassification::HAZARD}};
const std::map<std::string, std::uint8_t> OBJECT_LABEL_TO_UINT_MAP = {
  {"UNKNOWN", ObjectClassification::UNKNOWN}, {"CAR", ObjectClassification::CAR},
  {"TRUCK", ObjectClassification::TRUCK},     {"BUS", ObjectClassification::BUS},
  {"TRAILER", ObjectClassification::TRAILER}, {"MOTORCYCLE", ObjectClassification::MOTORCYCLE},
  {"BICYCLE", ObjectClassification::BICYCLE}, {"PEDESTRIAN", ObjectClassification::PEDESTRIAN},
  {"ANIMAL", ObjectClassification::ANIMAL}};

// The object fields a radar has to provide for the conversion to make sense.
const std::vector<std::string> REQUIRED_FIELDS = {
  "existence_probability", "position_x",     "position_y", "velocity_x", "velocity_y",
  "acceleration_x",        "acceleration_y", "orientation"};

float mask_cov_value(double value)
{
  return static_cast<float>(
    value == autoware_sensing_msgs::msg::RadarObject::INVALID_COV_VALUE ? 0.0 : value);
}

void radar_cov_to_detection_pose_cov(
  const std::array<float, 6> & radar_pose_cov, const double orientation_std,
  const bool orientation_std_available, std::array<double, 36> & pose_cov)
{
  pose_cov[DETECTION_COV_IDX::X_X] = mask_cov_value(radar_pose_cov[RADAR_COV_IDX::X_X]);
  pose_cov[DETECTION_COV_IDX::X_Y] = mask_cov_value(radar_pose_cov[RADAR_COV_IDX::X_Y]);
  pose_cov[DETECTION_COV_IDX::Y_X] = mask_cov_value(radar_pose_cov[RADAR_COV_IDX::X_Y]);
  pose_cov[DETECTION_COV_IDX::Y_Y] = mask_cov_value(radar_pose_cov[RADAR_COV_IDX::Y_Y]);

  if (orientation_std_available) {
    pose_cov[DETECTION_COV_IDX::YAW_YAW] = static_cast<double>(orientation_std * orientation_std);
  }
}

void radar_cov_to_detection_twist_cov(
  const std::array<float, 6> & radar_twist_cov, const float yaw, const float yaw_rate_std,
  const bool yaw_rate_std_available, std::array<double, 36> & twist_cov)
{
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);

  const float xx = mask_cov_value(radar_twist_cov[RADAR_COV_IDX::X_X]);
  const float xy = mask_cov_value(radar_twist_cov[RADAR_COV_IDX::X_Y]);
  const float yy = mask_cov_value(radar_twist_cov[RADAR_COV_IDX::Y_Y]);

  twist_cov[DETECTION_COV_IDX::X_X] =
    static_cast<double>(xx * c * c + yy * s * s + 2.f * xy * s * c);

  twist_cov[DETECTION_COV_IDX::X_Y] = static_cast<double>((yy - xx) * s * c + xy * (c * c - s * s));
  twist_cov[DETECTION_COV_IDX::Y_X] = twist_cov[DETECTION_COV_IDX::X_Y];

  twist_cov[DETECTION_COV_IDX::Y_Y] =
    static_cast<double>(xx * s * s + yy * c * c - 2.f * xy * s * c);

  twist_cov[DETECTION_COV_IDX::Y_Z] = 0.0;

  if (yaw_rate_std_available) {
    twist_cov[DETECTION_COV_IDX::YAW_YAW] = static_cast<double>(yaw_rate_std * yaw_rate_std);
  }
}

void radar_cov_to_detection_acceleration_cov(
  const std::array<float, 6> & radar_acceleration_cov, const float yaw,
  std::array<double, 36> & acceleration_cov)
{
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);

  const float xx = mask_cov_value(radar_acceleration_cov[RADAR_COV_IDX::X_X]);
  const float xy = mask_cov_value(radar_acceleration_cov[RADAR_COV_IDX::X_Y]);
  const float yy = mask_cov_value(radar_acceleration_cov[RADAR_COV_IDX::Y_Y]);

  acceleration_cov[DETECTION_COV_IDX::X_X] =
    static_cast<double>(xx * c * c + yy * s * s + 2.f * xy * s * c);

  acceleration_cov[DETECTION_COV_IDX::X_Y] =
    static_cast<double>((yy - xx) * s * c + xy * (c * c - s * s));
  acceleration_cov[DETECTION_COV_IDX::Y_X] = acceleration_cov[DETECTION_COV_IDX::X_Y];

  acceleration_cov[DETECTION_COV_IDX::Y_Y] =
    static_cast<double>(xx * s * s + yy * c * c - 2.f * xy * s * c);

  acceleration_cov[DETECTION_COV_IDX::Y_Z] = 0.0;
}

template <typename ObjectType>
void populate_common_fields(
  const autoware_sensing_msgs::msg::RadarObject & input_object, ObjectType & output_object,
  const float yaw, const RadarObjectsAdapterParams & params,
  const RadarFieldAvailability & availability)
{
  output_object.existence_probability = input_object.existence_probability;

  auto & output_shape = output_object.shape;
  output_shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  output_shape.dimensions.x = availability.size_x() ? input_object.size.x : params.default_size_x;
  output_shape.dimensions.y = availability.size_y() ? input_object.size.y : params.default_size_y;
  output_shape.dimensions.z = availability.size_z() ? input_object.size.z : params.default_size_z;

  auto & output_pose = output_object.kinematics.pose_with_covariance.pose;
  output_pose.position.x = input_object.position.x;
  output_pose.position.y = input_object.position.y;
  output_pose.position.z =
    availability.position_z() ? input_object.position.z : params.default_position_z;
  output_pose.orientation = autoware_utils_geometry::create_quaternion_from_yaw(yaw);

  radar_cov_to_detection_pose_cov(
    input_object.position_covariance, input_object.orientation_std, availability.orientation_std(),
    output_object.kinematics.pose_with_covariance.covariance);

  auto & output_twist = output_object.kinematics.twist_with_covariance.twist;
  output_twist.linear.x =
    std::cos(yaw) * input_object.velocity.x + std::sin(yaw) * input_object.velocity.y;
  output_twist.linear.y =
    -std::sin(yaw) * input_object.velocity.x + std::cos(yaw) * input_object.velocity.y;
  output_twist.linear.z =
    availability.velocity_z() ? input_object.velocity.z : params.default_velocity_z;
  output_twist.angular.z = input_object.orientation_rate;

  radar_cov_to_detection_twist_cov(
    input_object.velocity_covariance, yaw, input_object.orientation_rate_std,
    availability.orientation_rate_std(), output_object.kinematics.twist_with_covariance.covariance);

  // Additional fields for TrackedObject
  if constexpr (std::is_same_v<ObjectType, autoware_perception_msgs::msg::TrackedObject>) {
    auto & output_acceleration = output_object.kinematics.acceleration_with_covariance.accel;
    output_acceleration.linear.x =
      std::cos(yaw) * input_object.acceleration.x + std::sin(yaw) * input_object.acceleration.y;
    output_acceleration.linear.y =
      -std::sin(yaw) * input_object.acceleration.x + std::cos(yaw) * input_object.acceleration.y;
    output_acceleration.linear.z =
      availability.acceleration_z() ? input_object.acceleration.z : params.default_acceleration_z;

    radar_cov_to_detection_acceleration_cov(
      input_object.acceleration_covariance, yaw,
      output_object.kinematics.acceleration_with_covariance.covariance);
  }
}

void populate_classifications(
  const std::vector<autoware_sensing_msgs::msg::RadarClassification> & input_classifications,
  const std::map<std::uint8_t, std::uint8_t> & classification_remap,
  std::vector<autoware_perception_msgs::msg::ObjectClassification> & output_classifications)
{
  for (const auto & input_classification : input_classifications) {
    // class remap based on policy defined in parameter
    if (classification_remap.count(input_classification.label)) {
      ObjectClassification output_classification;
      output_classification.label = classification_remap.at(input_classification.label);
      output_classification.probability = input_classification.probability;
      output_classifications.push_back(output_classification);
    } else {
      // if no remap rule matched, set UNKNOWN
      ObjectClassification output_classification;
      output_classification.label = ObjectClassification::UNKNOWN;
      output_classification.probability = input_classification.probability;
      output_classifications.push_back(output_classification);
    }
  }
}
}  // namespace

RadarFieldAvailability::UpdateResult RadarFieldAvailability::update(
  const autoware_sensing_msgs::msg::RadarInfo & radar_info)
{
  for (const auto & field_info : radar_info.object_fields_info) {
    field_info_map_[field_info.field_name.data] = field_info;
  }

  UpdateResult result;
  for (const auto & field : REQUIRED_FIELDS) {
    if (field_info_map_.find(field) == field_info_map_.end()) {
      result.missing_required_fields.push_back(field);
    }
  }
  result.valid = result.missing_required_fields.empty();
  valid_ = result.valid;

  if (!valid_) {
    return result;
  }

  position_z_ = field_info_map_.count("position_z") > 0;
  velocity_z_ = field_info_map_.count("velocity_z") > 0;
  acceleration_z_ = field_info_map_.count("acceleration_z") > 0;
  size_x_ = field_info_map_.count("size_x") > 0;
  size_y_ = field_info_map_.count("size_y") > 0;
  size_z_ = field_info_map_.count("size_z") > 0;

  orientation_std_ = field_info_map_.count("orientation_std") > 0;
  orientation_rate_std_ = field_info_map_.count("orientation_rate_std") > 0;

  return result;
}

ClassificationRemap make_classification_remap(
  const std::map<std::string, std::string> & remap_by_name)
{
  ClassificationRemap remap;
  for (const auto & kv : remap_by_name) {
    const std::string & radar_label = kv.first;        // e.g. "CAR"
    const std::string & perception_label = kv.second;  // e.g. "TRUCK"

    // Radar string → uint8
    uint8_t radar_id = RADAR_LABEL_TO_UINT_MAP.at(radar_label);

    // Perception string → uint8
    uint8_t perception_id = ObjectClassification::UNKNOWN;
    auto it = OBJECT_LABEL_TO_UINT_MAP.find(perception_label);
    if (it != OBJECT_LABEL_TO_UINT_MAP.end()) {
      perception_id = it->second;
    } else {
      remap.unknown_perception_labels.emplace_back(radar_label, perception_label);
    }

    remap.label_map[radar_id] = perception_id;
  }
  return remap;
}

RadarObjectsAdapter::RadarObjectsAdapter(
  const RadarObjectsAdapterParams & params,
  std::map<std::uint8_t, std::uint8_t> classification_remap, const UuidSourceId & uuid_source_id)
: params_(params),
  classification_remap_(std::move(classification_remap)),
  uuid_source_id_(uuid_source_id)
{
}

RadarFieldAvailability::UpdateResult RadarObjectsAdapter::update_radar_info(
  const autoware_sensing_msgs::msg::RadarInfo & radar_info)
{
  return availability_.update(radar_info);
}

RadarObjectsAdapter::Result RadarObjectsAdapter::convert(
  const autoware_sensing_msgs::msg::RadarObjects & input_msg) const
{
  Result result;
  if (!availability_.valid()) {
    result.outcome = Outcome::NoValidRadarInfo;
    return result;
  }

  result.outcome = Outcome::Converted;
  result.detections = to_detected_objects(input_msg);
  result.tracks = to_tracked_objects(input_msg);
  return result;
}

autoware_perception_msgs::msg::DetectedObjects RadarObjectsAdapter::to_detected_objects(
  const autoware_sensing_msgs::msg::RadarObjects & input_msg) const
{
  autoware_perception_msgs::msg::DetectedObjects output_msg;

  output_msg.header = input_msg.header;
  output_msg.objects.reserve(input_msg.objects.size());

  for (const auto & input_object : input_msg.objects) {
    autoware_perception_msgs::msg::DetectedObject output_object;

    // Populate common fields
    const auto & yaw = input_object.orientation;
    populate_common_fields(input_object, output_object, yaw, params_, availability_);

    // Set flags for kinematics
    output_object.kinematics.has_position_covariance = true;
    output_object.kinematics.orientation_availability =
      autoware_perception_msgs::msg::DetectedObjectKinematics::AVAILABLE;
    output_object.kinematics.has_twist = true;
    output_object.kinematics.has_twist_covariance = true;

    // Set classification
    populate_classifications(
      input_object.classifications, classification_remap_, output_object.classification);

    output_msg.objects.push_back(output_object);
  }

  return output_msg;
}

autoware_perception_msgs::msg::TrackedObjects RadarObjectsAdapter::to_tracked_objects(
  const autoware_sensing_msgs::msg::RadarObjects & input_msg) const
{
  autoware_perception_msgs::msg::TrackedObjects output_msg;

  output_msg.header = input_msg.header;
  output_msg.objects.reserve(input_msg.objects.size());

  for (const auto & input_object : input_msg.objects) {
    autoware_perception_msgs::msg::TrackedObject output_object;

    output_object.object_id.uuid[0] = static_cast<uint8_t>((input_object.object_id >> 0) & 0xFF);
    output_object.object_id.uuid[1] = static_cast<uint8_t>((input_object.object_id >> 8) & 0xFF);
    output_object.object_id.uuid[2] = static_cast<uint8_t>((input_object.object_id >> 16) & 0xFF);
    output_object.object_id.uuid[3] = static_cast<uint8_t>((input_object.object_id >> 24) & 0xFF);

    for (std::size_t i = 4; i < output_object.object_id.uuid.size(); ++i) {
      if (i - 4 < uuid_source_id_.size()) {
        output_object.object_id.uuid[i] = uuid_source_id_[i - 4];
      } else {
        output_object.object_id.uuid[i] = 0;
      }
    }

    // Populate common fields
    const auto & yaw = input_object.orientation;
    populate_common_fields(input_object, output_object, yaw, params_, availability_);

    // Set flags for kinematics
    output_object.kinematics.orientation_availability =
      autoware_perception_msgs::msg::TrackedObjectKinematics::AVAILABLE;
    output_object.kinematics.is_stationary =
      input_object.movement_status !=
      autoware_sensing_msgs::msg::RadarObject::MOVEMENT_STATUS_DYNAMIC;

    // Populate classification
    populate_classifications(
      input_object.classifications, classification_remap_, output_object.classification);

    output_msg.objects.push_back(output_object);
  }

  return output_msg;
}

}  // namespace autoware::radar_objects_adapter
