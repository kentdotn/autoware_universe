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

// The conversion has three parts, composed by RadarObjectsAdapter: the classification remap
// (ClassificationRemapper), the conversion of one radar object into a tracked object
// (TrackedObjectConverter, parameterized by which fields the radar provides and by how a tracked
// object's id is made), and the derivation of the detected object from the tracked one.

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/tracked_object.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_sensing_msgs/msg/radar_classification.hpp>
#include <autoware_sensing_msgs/msg/radar_info.hpp>
#include <autoware_sensing_msgs/msg/radar_object.hpp>
#include <autoware_sensing_msgs/msg/radar_objects.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// ---------------------------------------------------------------------------------------------
// Classification remap
// ---------------------------------------------------------------------------------------------

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

// The remap every radar needs, whatever it reports: the labels the perception pipeline has no
// use for (HAZARD, OVER_DRIVABLE, UNDER_DRIVABLE) become UNKNOWN.
ClassificationRemap perception_friendly_classification_remap();

// Turns the classifications of a radar object into perception classifications: one entry per
// radar entry, in order, with the probability copied and the label looked up in one table. The
// table is the perception-friendly remap with the sensor-dependent one laid over it (the
// classification_remap parameters, which correct what the radar in use is known to report
// unreliably), so that the sensor's entry wins where both say something. A label neither
// mentions becomes UNKNOWN.
class ClassificationRemapper
{
public:
  ClassificationRemapper(
    const ClassificationRemap & perception_friendly, const ClassificationRemap & sensor_dependent);

  std::vector<autoware_perception_msgs::msg::ObjectClassification> operator()(
    const std::vector<autoware_sensing_msgs::msg::RadarClassification> & classifications) const;

  const ClassificationRemap & combined() const { return combined_; }

private:
  ClassificationRemap combined_;
};

// ---------------------------------------------------------------------------------------------
// Tracked object conversion
// ---------------------------------------------------------------------------------------------

// How a tracked object's id is made from the radar's 32-bit object id. A function, so that the
// conversion can be given another way of making ids.
using GenerateObjectUUID =
  std::function<unique_identifier_msgs::msg::UUID::_uuid_type(std::uint32_t object_id)>;

// The way the node makes them: the id, least significant byte first, then the hash of the input
// topic name, least significant byte first, then zeros. The hash tells the tracks of one radar
// from those of another when they are merged downstream.
class ObjectUUIDGenerator
{
public:
  explicit ObjectUUIDGenerator(const std::string & topic_name);

  unique_identifier_msgs::msg::UUID::_uuid_type operator()(std::uint32_t object_id) const;

private:
  std::array<std::uint8_t, sizeof(std::size_t)> topic_hash_code_;
};

// The values used for the object fields a radar does not provide.
struct RadarObjectsAdapterParams
{
  double default_position_z{0.0};
  double default_velocity_z{0.0};
  double default_acceleration_z{0.0};
  double default_size_x{0.0};
  double default_size_y{0.0};
  double default_size_z{0.0};
};

// Which optional object fields the radar provides, as its radar info declared them. A field that
// is not provided is filled from the parameters (the six that have one) or left out (the two
// standard deviations, whose variances then stay zero).
struct RadarFieldAvailability
{
  bool position_z{false};
  bool velocity_z{false};
  bool acceleration_z{false};
  bool size_x{false};
  bool size_y{false};
  bool size_z{false};

  bool orientation_std{false};
  bool orientation_rate_std{false};
};

// Converts one radar object into a tracked object: everything the radar reports about it, with
// the fields the radar does not provide filled from the parameters, the velocity and the
// acceleration rotated into the object's frame, and the id made by the given function. The
// classification is not set here; the remapper supplies it.
class TrackedObjectConverter
{
public:
  TrackedObjectConverter(
    const RadarObjectsAdapterParams & params, const RadarFieldAvailability & availability,
    GenerateObjectUUID generate_uuid);

  autoware_perception_msgs::msg::TrackedObject operator()(
    const autoware_sensing_msgs::msg::RadarObject & input_object) const;

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

  RadarObjectsAdapterParams params_;
  RadarFieldAvailability availability_;
  GenerateObjectUUID generate_uuid_;
};

// ---------------------------------------------------------------------------------------------
// The adapter
// ---------------------------------------------------------------------------------------------

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
    std::vector<std::pair<std::string, double>> defaulted_fields;

    bool valid() const { return missing_required_fields.empty(); }
  };

  RadarObjectsAdapter(
    const RadarObjectsAdapterParams & params, const ClassificationRemap & classification_remap,
    const std::string & topic_name);

  RadarInfoResult update_radar_info(const autoware_sensing_msgs::msg::RadarInfo & radar_info_msg);

  Result convert(const autoware_sensing_msgs::msg::RadarObjects & input_msg) const;

private:
  RadarObjectsAdapterParams params_;
  GenerateObjectUUID generate_uuid_;
  ClassificationRemapper classification_remapper_;

  std::unordered_map<std::string, autoware_sensing_msgs::msg::RadarFieldInfo> field_info_map_;
  std::vector<std::string> required_attributes_;

  // Set once a radar info has declared every required field; converts nothing before.
  std::optional<TrackedObjectConverter> tracked_object_converter_;
};

}  // namespace autoware::radar_objects_adapter

#endif  // RADAR_OBJECTS_ADAPTER_HPP_
