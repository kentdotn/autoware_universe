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

// The conversion of radar objects into perception objects, free of rclcpp. The node
// (radar_objects_adapter_node.hpp) owns the topics, the parameters and the logging, and hands the
// values it read to the types below.

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
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

// Which object fields a radar provides, learned from its radar info messages. The fields declared
// by successive messages accumulate. The radar is usable once every required field has been
// declared; the optional fields decide whether a value is copied from the radar object or taken
// from the parameters.
class RadarFieldAvailability
{
public:
  struct UpdateResult
  {
    bool valid{false};
    // The required fields that no radar info has declared so far, in the order they are required.
    std::vector<std::string> missing_required_fields;
  };

  UpdateResult update(const autoware_sensing_msgs::msg::RadarInfo & radar_info);

  bool valid() const { return valid_; }

  bool position_z() const { return position_z_; }
  bool velocity_z() const { return velocity_z_; }
  bool acceleration_z() const { return acceleration_z_; }
  bool size_x() const { return size_x_; }
  bool size_y() const { return size_y_; }
  bool size_z() const { return size_z_; }
  bool orientation_std() const { return orientation_std_; }
  bool orientation_rate_std() const { return orientation_rate_std_; }

private:
  std::unordered_map<std::string, autoware_sensing_msgs::msg::RadarFieldInfo> field_info_map_;

  bool valid_{false};

  bool position_z_{false};
  bool velocity_z_{false};
  bool acceleration_z_{false};
  bool size_x_{false};
  bool size_y_{false};
  bool size_z_{false};
  bool orientation_std_{false};
  bool orientation_rate_std_{false};
};

// The classification remap as label ids, built from the radar label -> perception label names of
// the parameters.
struct ClassificationRemap
{
  std::map<std::uint8_t, std::uint8_t> label_map;
  // (radar label, perception label) pairs whose perception label is not a known label name. Their
  // radar label maps to UNKNOWN.
  std::vector<std::pair<std::string, std::string>> unknown_perception_labels;
};

// `remap_by_name` must have RadarClassification label names as keys.
ClassificationRemap make_classification_remap(
  const std::map<std::string, std::string> & remap_by_name);

// Converts radar objects into detected objects and into tracked objects. Learns which fields the
// radar provides from its radar info messages; until a valid one has arrived, nothing is converted.
class RadarObjectsAdapter
{
public:
  // The bytes of a tracked object's UUID that follow the radar's object id. They tell the tracks
  // of one radar from those of another.
  using UuidSourceId = std::array<std::uint8_t, sizeof(std::size_t)>;

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

  RadarObjectsAdapter(
    const RadarObjectsAdapterParams & params,
    std::map<std::uint8_t, std::uint8_t> classification_remap, const UuidSourceId & uuid_source_id);

  RadarFieldAvailability::UpdateResult update_radar_info(
    const autoware_sensing_msgs::msg::RadarInfo & radar_info);

  Result convert(const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;

  const RadarObjectsAdapterParams & params() const { return params_; }
  const RadarFieldAvailability & availability() const { return availability_; }

private:
  autoware_perception_msgs::msg::DetectedObjects to_detected_objects(
    const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;
  autoware_perception_msgs::msg::TrackedObjects to_tracked_objects(
    const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;

  RadarObjectsAdapterParams params_;
  std::map<std::uint8_t, std::uint8_t> classification_remap_;
  UuidSourceId uuid_source_id_;
  RadarFieldAvailability availability_;
};

}  // namespace autoware::radar_objects_adapter

#endif  // RADAR_OBJECTS_ADAPTER_HPP_
