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

#include <autoware/object_recognition_utils/conversion.hpp>
#include <autoware_utils_geometry/geometry.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace autoware::radar_objects_adapter
{

// Maps for classification remapping
const std::map<std::string, RadarClassification::_label_type> RADAR_LABEL_TO_UINT_MAP = {
  {"UNKNOWN", RadarClassification::UNKNOWN}, {"CAR", RadarClassification::CAR},
  {"TRUCK", RadarClassification::TRUCK},     {"BUS", RadarClassification::BUS},
  {"TRAILER", RadarClassification::TRAILER}, {"MOTORCYCLE", RadarClassification::MOTORCYCLE},
  {"BICYCLE", RadarClassification::BICYCLE}, {"PEDESTRIAN", RadarClassification::PEDESTRIAN},
  {"ANIMAL", RadarClassification::ANIMAL},   {"HAZARD", RadarClassification::HAZARD}};
const std::map<std::string, ObjectClassification::_label_type> OBJECT_LABEL_TO_UINT_MAP = {
  {"UNKNOWN", ObjectClassification::UNKNOWN}, {"CAR", ObjectClassification::CAR},
  {"TRUCK", ObjectClassification::TRUCK},     {"BUS", ObjectClassification::BUS},
  {"TRAILER", ObjectClassification::TRAILER}, {"MOTORCYCLE", ObjectClassification::MOTORCYCLE},
  {"BICYCLE", ObjectClassification::BICYCLE}, {"PEDESTRIAN", ObjectClassification::PEDESTRIAN},
  {"ANIMAL", ObjectClassification::ANIMAL}};

ClassificationRemap make_classification_remap_from_string_pair(
  const std::map<std::string, std::string> & classification_remap_str,
  std::map<std::string, std::string> & unknown_perception_labels)
{
  ClassificationRemap classification_remap;

  for (const auto & [radar_label, perception_label] : classification_remap_str) {
    // Radar string → uint8
    auto radar_id = RADAR_LABEL_TO_UINT_MAP.at(radar_label);

    // Perception string → uint8
    auto perception_id = ObjectClassification::UNKNOWN;
    auto it = OBJECT_LABEL_TO_UINT_MAP.find(perception_label);
    if (it != OBJECT_LABEL_TO_UINT_MAP.end()) {
      perception_id = it->second;
    } else {
      unknown_perception_labels[radar_label] = perception_label;
    }

    classification_remap[radar_id] = perception_id;
  }

  return classification_remap;
}

ObjectUUIDGenerator::ObjectUUIDGenerator(const std::string & topic_name)
{
  const std::size_t hash_code = std::hash<std::string>{}(topic_name);
  for (std::size_t i = 0; i < sizeof(std::size_t); ++i) {
    topic_hash_code_[i] = static_cast<std::uint8_t>((hash_code >> (i * 8)) & 0xFF);
  }
}

unique_identifier_msgs::msg::UUID::_uuid_type ObjectUUIDGenerator::operator()(
  const std::uint32_t object_id) const
{
  unique_identifier_msgs::msg::UUID::_uuid_type uuid;

  uuid[0] = static_cast<uint8_t>((object_id >> 0) & 0xFF);
  uuid[1] = static_cast<uint8_t>((object_id >> 8) & 0xFF);
  uuid[2] = static_cast<uint8_t>((object_id >> 16) & 0xFF);
  uuid[3] = static_cast<uint8_t>((object_id >> 24) & 0xFF);

  for (std::size_t i = 4; i < uuid.size(); ++i) {
    if (i - 4 < topic_hash_code_.size()) {
      uuid[i] = topic_hash_code_[i - 4];
    } else {
      uuid[i] = 0;
    }
  }

  return uuid;
}

namespace
{

float mask_cov_value(double value)
{
  return static_cast<float>(
    value == autoware_sensing_msgs::msg::RadarObject::INVALID_COV_VALUE ? 0.0 : value);
}

using DETECTION_COV_IDX = autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
using RADAR_COV_IDX = autoware_utils_geometry::xyz_upper_covariance_index::XYZ_UPPER_COV_IDX;

// The variance a standard deviation the radar may or may not provide contributes to a
// covariance: its square when the field is available, nothing otherwise.
std::optional<double> variance_if_available(const bool available, const float std_dev)
{
  if (!available) {
    return std::nullopt;
  }
  return static_cast<double>(std_dev * std_dev);
}

// A 6x6 pose covariance whose x/y block is copied from the radar's upper triangle as it is,
// with the yaw variance if there is one. Every other entry is zero.
std::array<double, 36> copied_xy_covariance(
  const std::array<float, 6> & radar_cov, const std::optional<double> & yaw_variance)
{
  std::array<double, 36> cov{};

  cov[DETECTION_COV_IDX::X_X] = mask_cov_value(radar_cov[RADAR_COV_IDX::X_X]);
  cov[DETECTION_COV_IDX::X_Y] = mask_cov_value(radar_cov[RADAR_COV_IDX::X_Y]);
  cov[DETECTION_COV_IDX::Y_X] = mask_cov_value(radar_cov[RADAR_COV_IDX::X_Y]);
  cov[DETECTION_COV_IDX::Y_Y] = mask_cov_value(radar_cov[RADAR_COV_IDX::Y_Y]);

  if (yaw_variance.has_value()) {
    cov[DETECTION_COV_IDX::YAW_YAW] = yaw_variance.value();
  }
  return cov;
}

// A 6x6 covariance whose x/y block is the radar's, rotated into the frame of an object facing
// `yaw`, with the yaw variance if there is one. Every other entry is zero. Serves the twist and
// the acceleration alike.
std::array<double, 36> rotated_xy_covariance(
  const std::array<float, 6> & radar_cov, const float yaw,
  const std::optional<double> & yaw_variance)
{
  std::array<double, 36> cov{};

  const float c = std::cos(yaw);
  const float s = std::sin(yaw);

  const float xx = mask_cov_value(radar_cov[RADAR_COV_IDX::X_X]);
  const float xy = mask_cov_value(radar_cov[RADAR_COV_IDX::X_Y]);
  const float yy = mask_cov_value(radar_cov[RADAR_COV_IDX::Y_Y]);

  cov[DETECTION_COV_IDX::X_X] = static_cast<double>(xx * c * c + yy * s * s + 2.f * xy * s * c);

  cov[DETECTION_COV_IDX::X_Y] = static_cast<double>((yy - xx) * s * c + xy * (c * c - s * s));
  cov[DETECTION_COV_IDX::Y_X] = cov[DETECTION_COV_IDX::X_Y];

  cov[DETECTION_COV_IDX::Y_Y] = static_cast<double>(xx * s * s + yy * c * c - 2.f * xy * s * c);

  if (yaw_variance.has_value()) {
    cov[DETECTION_COV_IDX::YAW_YAW] = yaw_variance.value();
  }
  return cov;
}

}  // namespace

ClassificationRemap perception_friendly_classification_remap()
{
  return ClassificationRemap{
    {RadarClassification::HAZARD, ObjectClassification::UNKNOWN},
    {RadarClassification::OVER_DRIVABLE, ObjectClassification::UNKNOWN},
    {RadarClassification::UNDER_DRIVABLE, ObjectClassification::UNKNOWN}};
}

ClassificationRemapper::ClassificationRemapper(
  const ClassificationRemap & perception_friendly, const ClassificationRemap & sensor_dependent)
: combined_(perception_friendly)
{
  for (const auto & [radar_label, perception_label] : sensor_dependent) {
    combined_[radar_label] = perception_label;
  }
}

std::vector<autoware_perception_msgs::msg::ObjectClassification> ClassificationRemapper::operator()(
  const std::vector<autoware_sensing_msgs::msg::RadarClassification> & classifications) const
{
  std::vector<ObjectClassification> output;
  output.reserve(classifications.size());
  for (const auto & classification : classifications) {
    ObjectClassification remapped;
    // class remap based on policy defined in parameter; if no remap rule matched, set UNKNOWN
    const auto it = combined_.find(classification.label);
    remapped.label = it != combined_.end() ? it->second : ObjectClassification::UNKNOWN;
    remapped.probability = classification.probability;
    output.push_back(remapped);
  }
  return output;
}

TrackedObjectConverter::TrackedObjectConverter(
  const RadarObjectsAdapterParams & params, const RadarFieldAvailability & availability,
  GenerateObjectUUID generate_uuid)
: params_(params), availability_(availability), generate_uuid_(std::move(generate_uuid))
{
}

autoware_perception_msgs::msg::TrackedObject TrackedObjectConverter::operator()(
  const autoware_sensing_msgs::msg::RadarObject & input_object) const
{
  autoware_perception_msgs::msg::TrackedObject output_object;

  output_object.object_id.set__uuid(generate_uuid_(input_object.object_id));

  output_object.existence_probability = input_object.existence_probability;

  auto & output_shape = output_object.shape;
  output_shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  output_shape.dimensions.x = availability_.size_x ? input_object.size.x : params_.default_size_x;
  output_shape.dimensions.y = availability_.size_y ? input_object.size.y : params_.default_size_y;
  output_shape.dimensions.z = availability_.size_z ? input_object.size.z : params_.default_size_z;

  const float yaw = input_object.orientation;

  auto & output_pose = output_object.kinematics.pose_with_covariance.pose;
  output_pose.position.x = input_object.position.x;
  output_pose.position.y = input_object.position.y;
  output_pose.position.z =
    availability_.position_z ? input_object.position.z : params_.default_position_z;
  output_pose.orientation = autoware_utils_geometry::create_quaternion_from_yaw(yaw);

  output_object.kinematics.pose_with_covariance.covariance = copied_xy_covariance(
    input_object.position_covariance,
    variance_if_available(availability_.orientation_std, input_object.orientation_std));

  auto & output_twist = output_object.kinematics.twist_with_covariance.twist;
  output_twist.linear.x =
    std::cos(yaw) * input_object.velocity.x + std::sin(yaw) * input_object.velocity.y;
  output_twist.linear.y =
    -std::sin(yaw) * input_object.velocity.x + std::cos(yaw) * input_object.velocity.y;
  output_twist.linear.z =
    availability_.velocity_z ? input_object.velocity.z : params_.default_velocity_z;
  output_twist.angular.z = input_object.orientation_rate;

  output_object.kinematics.twist_with_covariance.covariance = rotated_xy_covariance(
    input_object.velocity_covariance, yaw,
    variance_if_available(availability_.orientation_rate_std, input_object.orientation_rate_std));

  auto & output_acceleration = output_object.kinematics.acceleration_with_covariance.accel;
  output_acceleration.linear.x =
    std::cos(yaw) * input_object.acceleration.x + std::sin(yaw) * input_object.acceleration.y;
  output_acceleration.linear.y =
    -std::sin(yaw) * input_object.acceleration.x + std::cos(yaw) * input_object.acceleration.y;
  output_acceleration.linear.z =
    availability_.acceleration_z ? input_object.acceleration.z : params_.default_acceleration_z;

  output_object.kinematics.acceleration_with_covariance.covariance =
    rotated_xy_covariance(input_object.acceleration_covariance, yaw, std::nullopt);

  // Set flags for kinematics
  output_object.kinematics.orientation_availability =
    autoware_perception_msgs::msg::TrackedObjectKinematics::AVAILABLE;
  output_object.kinematics.is_stationary =
    input_object.movement_status !=
    autoware_sensing_msgs::msg::RadarObject::MOVEMENT_STATUS_DYNAMIC;

  return output_object;
}

RadarObjectsAdapter::RadarObjectsAdapter(
  const RadarObjectsAdapterParams & params, const ClassificationRemap & classification_remap,
  const std::string & topic_name)
: params_(params),
  generate_uuid_(ObjectUUIDGenerator(topic_name)),
  classification_remapper_(perception_friendly_classification_remap(), classification_remap)
{
  required_attributes_ = {
    "existence_probability", "position_x",     "position_y", "velocity_x", "velocity_y",
    "acceleration_x",        "acceleration_y", "orientation"};
}

RadarObjectsAdapter::RadarInfoResult RadarObjectsAdapter::update_radar_info(
  const autoware_sensing_msgs::msg::RadarInfo & radar_info_msg)
{
  for (const auto & field_info : radar_info_msg.object_fields_info) {
    field_info_map_[field_info.field_name.data] = field_info;
  }

  RadarInfoResult result;
  for (const auto & attribute : required_attributes_) {
    if (field_info_map_.find(attribute) == field_info_map_.end()) {
      result.missing_required_fields.push_back(attribute);
    }
  }
  if (!result.valid()) {
    tracked_object_converter_.reset();
    return result;
  }

  RadarFieldAvailability availability;
  availability.position_z = field_info_map_.count("position_z") > 0;
  availability.velocity_z = field_info_map_.count("velocity_z") > 0;
  availability.acceleration_z = field_info_map_.count("acceleration_z") > 0;
  availability.size_x = field_info_map_.count("size_x") > 0;
  availability.size_y = field_info_map_.count("size_y") > 0;
  availability.size_z = field_info_map_.count("size_z") > 0;

  availability.orientation_std = field_info_map_.count("orientation_std") > 0;
  availability.orientation_rate_std = field_info_map_.count("orientation_rate_std") > 0;

  tracked_object_converter_.emplace(params_, availability, generate_uuid_);

  // The fields filled from a parameter, in the order the node has always warned about them.
  // orientation_std and orientation_rate_std have no parameter: without them the yaw variances
  // stay zero, and that has never been warned about.
  if (!availability.position_z) {
    result.defaulted_fields.emplace_back("position_z", params_.default_position_z);
  }
  if (!availability.velocity_z) {
    result.defaulted_fields.emplace_back("velocity_z", params_.default_velocity_z);
  }
  if (!availability.acceleration_z) {
    result.defaulted_fields.emplace_back("acceleration_z", params_.default_acceleration_z);
  }
  if (!availability.size_x) {
    result.defaulted_fields.emplace_back("size_x", params_.default_size_x);
  }
  if (!availability.size_y) {
    result.defaulted_fields.emplace_back("size_y", params_.default_size_y);
  }
  if (!availability.size_z) {
    result.defaulted_fields.emplace_back("size_z", params_.default_size_z);
  }
  return result;
}

RadarObjectsAdapter::Result RadarObjectsAdapter::convert(
  const autoware_sensing_msgs::msg::RadarObjects & input_msg) const
{
  if (!tracked_object_converter_.has_value()) {
    return {Outcome::NoValidRadarInfo, {}, {}};
  }

  autoware_perception_msgs::msg::TrackedObjects tracks;
  tracks.header = input_msg.header;
  tracks.objects.reserve(input_msg.objects.size());

  autoware_perception_msgs::msg::DetectedObjects detections;
  detections.header = input_msg.header;
  detections.objects.reserve(input_msg.objects.size());

  // A track is a detection plus its identity, its acceleration and whether it stands still, so
  // the track is built first and the detection is what is left of it.
  for (const auto & input_object : input_msg.objects) {
    autoware_perception_msgs::msg::TrackedObject track = (*tracked_object_converter_)(input_object);
    track.classification = classification_remapper_(input_object.classifications);

    detections.objects.push_back(autoware::object_recognition_utils::toDetectedObject(track));
    tracks.objects.push_back(std::move(track));
  }

  return {Outcome::Converted, std::move(detections), std::move(tracks)};
}

}  // namespace autoware::radar_objects_adapter
