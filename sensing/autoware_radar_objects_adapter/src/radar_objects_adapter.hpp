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

#ifndef RADAR_OBJECTS_ADAPTER_HPP_
#define RADAR_OBJECTS_ADAPTER_HPP_

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_sensing_msgs/msg/radar_classification.hpp>
#include <autoware_sensing_msgs/msg/radar_info.hpp>
#include <autoware_sensing_msgs/msg/radar_objects.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::radar_objects_adapter
{

using RadarClassification = autoware_sensing_msgs::msg::RadarClassification;
using ObjectClassification = autoware_perception_msgs::msg::ObjectClassification;

// The classification remap as label ids, built from the radar label -> perception label names of
// the parameters.
using ClassificationRemap =
  std::map<RadarClassification::_label_type, ObjectClassification::_label_type>;

// `classification_remap_str` must have RadarClassification label names as keys. A value that is
// not an ObjectClassification label name maps that radar label to UNKNOWN and is reported in
// `unknown_perception_labels` (radar label -> the unknown perception label), for the node to warn
// about; the keys are the radar labels so that every such entry is reported once.
ClassificationRemap make_classification_remap_from_string_pair(
  const std::map<std::string, std::string> & classification_remap_str,
  std::map<std::string, std::string> & unknown_perception_labels);

// The values used for the object fields a radar does not provide.
struct RadarObjectsAdapterParams
{
  float default_position_z{0.0f};
  float default_velocity_z{0.0f};
  float default_acceleration_z{0.0f};
  float default_size_x{0.0f};
  float default_size_y{0.0f};
  float default_size_z{0.0f};
};

// Converts radar objects into detected objects and into tracked objects. Learns which fields the
// radar provides from its radar info messages; until a valid one has arrived, nothing is converted.
class RadarObjectsAdapter
{
public:
  enum class Outcome {
    NoValidRadarInfo,  ///< no radar info with all required fields yet; the objects were dropped
    Converted,         ///< see Result::detections and Result::tracks
  };

  // What a radar objects message produced, as the messages to publish. What is empty is not
  // published; `outcome` is there to be logged.
  struct Result
  {
    Outcome outcome;
    std::optional<autoware_perception_msgs::msg::DetectedObjects> detections;
    std::optional<autoware_perception_msgs::msg::TrackedObjects> tracks;
  };

  // What the adapter decided from a radar info: whether it can convert at all, and which optional
  // fields it will fill from a parameter instead of the object. The node logs this; the adapter
  // is where the field -> parameter pairing lives.
  struct RadarInfoResult
  {
    // The required fields that no radar info has declared so far, in the order they are
    // required. Empty when the radar info is valid.
    std::vector<std::string> missing_required_fields;
    // The optional fields the radar does not provide, each with the parameter value used in its
    // place. Empty while the radar info is not valid: nothing is decided from an incomplete one.
    std::vector<std::pair<std::string, float>> defaulted_fields;

    bool valid() const { return missing_required_fields.empty(); }
  };

  RadarObjectsAdapter(
    const RadarObjectsAdapterParams & params, const ClassificationRemap & classification_remap,
    const std::string & topic_name);

  RadarInfoResult update_radar_info(const autoware_sensing_msgs::msg::RadarInfo & radar_info_msg);

  Result convert(const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;

  const RadarObjectsAdapterParams & params() const { return params_; }

  bool valid_radar_info() const { return valid_radar_info_; }

  bool position_z_available() const { return position_z_available_; }
  bool velocity_z_available() const { return velocity_z_available_; }
  bool acceleration_z_available() const { return acceleration_z_available_; }
  bool size_x_available() const { return size_x_available_; }
  bool size_y_available() const { return size_y_available_; }
  bool size_z_available() const { return size_z_available_; }

  bool orientation_std_available() const { return orientation_std_available_; }
  bool orientation_rate_std_available() const { return orientation_rate_std_available_; }

private:
  void radar_cov_to_detection_pose_cov(
    const std::array<float, 6> & radar_pose_cov, const double orientation_std,
    std::array<double, 36> & pose_cov) const;

  void radar_cov_to_detection_twist_cov(
    const std::array<float, 6> & radar_twist_cov, const float yaw, const float yaw_rate_std,
    std::array<double, 36> & twist_cov) const;

  void radar_cov_to_detection_acceleration_cov(
    const std::array<float, 6> & radar_acceleration_cov, const float yaw,
    std::array<double, 36> & acceleration_cov) const;

  template <typename ObjectType>
  void populate_common_fields(
    const autoware_sensing_msgs::msg::RadarObject & input_object, ObjectType & output_object,
    const float yaw) const;

  void populate_classifications(
    const std::vector<autoware_sensing_msgs::msg::RadarClassification> & input_classifications,
    std::vector<autoware_perception_msgs::msg::ObjectClassification> & output_classifications)
    const;

  autoware_perception_msgs::msg::DetectedObjects to_detected_objects(
    const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;
  autoware_perception_msgs::msg::TrackedObjects to_tracked_objects(
    const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;

  RadarObjectsAdapterParams params_;

  std::unordered_map<std::string, autoware_sensing_msgs::msg::RadarFieldInfo> field_info_map_;

  bool valid_radar_info_{false};

  std::vector<std::string> required_attributes_;

  bool position_z_available_{false};
  bool velocity_z_available_{false};
  bool acceleration_z_available_{false};
  bool size_x_available_{false};
  bool size_y_available_{false};
  bool size_z_available_{false};

  bool orientation_std_available_{false};
  bool orientation_rate_std_available_{false};

  std::map<std::uint8_t, std::uint8_t> classification_remap_;

  std::array<std::uint8_t, sizeof(std::size_t)> topic_hash_code_;
};

}  // namespace autoware::radar_objects_adapter

#endif  // RADAR_OBJECTS_ADAPTER_HPP_
