// Copyright 2025 The Autoware Contributors
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

#include "radar_objects_adapter_node.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace autoware::radar_objects_adapter
{

RadarObjectsAdapterNode::RadarObjectsAdapterNode(const rclcpp::NodeOptions & options)
: Node("radar_objects_adapter", options)
{
  radar_objects_sub_ = this->create_subscription<autoware_sensing_msgs::msg::RadarObjects>(
    "~/input/objects", rclcpp::SensorDataQoS(),
    std::bind(&RadarObjectsAdapterNode::objects_callback, this, std::placeholders::_1));

  radar_info_sub_ = this->create_subscription<autoware_sensing_msgs::msg::RadarInfo>(
    "~/input/radar_info", rclcpp::SensorDataQoS(),
    std::bind(&RadarObjectsAdapterNode::radar_info_callback, this, std::placeholders::_1));

  detections_pub_ = this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
    "~/output/detections", rclcpp::QoS(10).reliable().transient_local());

  tracks_pub_ = this->create_publisher<autoware_perception_msgs::msg::TrackedObjects>(
    "~/output/tracks", rclcpp::QoS(10).reliable().transient_local());

  RadarObjectsAdapterParams params;
  params.default_position_z = this->declare_parameter<float>("default_position_z");
  params.default_velocity_z = this->declare_parameter<float>("default_velocity_z");
  params.default_acceleration_z = this->declare_parameter<float>("default_acceleration_z");

  params.default_size_x = this->declare_parameter<float>("default_size_x");
  params.default_size_y = this->declare_parameter<float>("default_size_y");
  params.default_size_z = this->declare_parameter<float>("default_size_z");

  // The tracks of this radar are told from those of another by the hash of the input topic name.
  std::size_t hash_code = std::hash<std::string>{}(radar_objects_sub_->get_topic_name());

  RadarObjectsAdapter::UuidSourceId uuid_source_id;
  for (std::size_t i = 0; i < sizeof(std::size_t); ++i) {
    uuid_source_id[i] = static_cast<std::uint8_t>((hash_code >> (i * 8)) & 0xFF);
  }

  // Load the classification remap policy: radar label -> perception label, as names in the
  // parameters and as label ids in the adapter.
  std::map<std::string, std::string> classification_remap_str;
  classification_remap_str["UNKNOWN"] =
    declare_parameter<std::string>("classification_remap.UNKNOWN", "UNKNOWN");
  classification_remap_str["CAR"] =
    declare_parameter<std::string>("classification_remap.CAR", "CAR");
  classification_remap_str["TRUCK"] =
    declare_parameter<std::string>("classification_remap.TRUCK", "TRUCK");
  classification_remap_str["MOTORCYCLE"] =
    declare_parameter<std::string>("classification_remap.MOTORCYCLE", "MOTORCYCLE");
  classification_remap_str["BICYCLE"] =
    declare_parameter<std::string>("classification_remap.BICYCLE", "BICYCLE");
  classification_remap_str["PEDESTRIAN"] =
    declare_parameter<std::string>("classification_remap.PEDESTRIAN", "PEDESTRIAN");
  classification_remap_str["ANIMAL"] =
    declare_parameter<std::string>("classification_remap.ANIMAL", "ANIMAL");
  classification_remap_str["HAZARD"] =
    declare_parameter<std::string>("classification_remap.HAZARD", "UNKNOWN");

  ClassificationRemap classification_remap = make_classification_remap(classification_remap_str);
  for (const auto & [radar_label, perception_label] :
       classification_remap.unknown_perception_labels) {
    RCLCPP_WARN(
      this->get_logger(),
      "classification_remap: invalid Perception label '%s' for radar '%s'. Using UNKNOWN.",
      perception_label.c_str(), radar_label.c_str());
  }

  adapter_.emplace(params, std::move(classification_remap.label_map), uuid_source_id);
}

void RadarObjectsAdapterNode::objects_callback(
  const autoware_sensing_msgs::msg::RadarObjects & objects_msg)
{
  const RadarObjectsAdapter::Result result = adapter_->convert(objects_msg);

  switch (result.outcome) {
    case RadarObjectsAdapter::Outcome::NoValidRadarInfo:
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "A Valid radar info message has not been received. Cannot convert radar objects.");
      break;
    case RadarObjectsAdapter::Outcome::Converted:
      break;
  }

  // publish both detections and tracks
  if (result.detections.has_value()) {
    auto output_msg_ptr = ALLOCATE_OUTPUT_MESSAGE_UNIQUE(detections_pub_);
    *output_msg_ptr = result.detections.value();
    detections_pub_->publish(std::move(output_msg_ptr));
  }
  if (result.tracks.has_value()) {
    auto output_msg_ptr = ALLOCATE_OUTPUT_MESSAGE_UNIQUE(tracks_pub_);
    *output_msg_ptr = result.tracks.value();
    tracks_pub_->publish(std::move(output_msg_ptr));
  }
}

void RadarObjectsAdapterNode::radar_info_callback(
  const autoware_sensing_msgs::msg::RadarInfo & radar_info_msg)
{
  const RadarFieldAvailability::UpdateResult result = adapter_->update_radar_info(radar_info_msg);

  if (!result.valid) {
    RCLCPP_ERROR_ONCE(
      get_logger(),
      "Radar info message is not valid. Some required attributes are missing. This radar may not "
      "be compatible with autoware");

    for (const auto & attribute : result.missing_required_fields) {
      RCLCPP_ERROR_ONCE(get_logger(), "\tMissing attribute: %s", attribute.c_str());
    }
    return;
  }

  const RadarFieldAvailability & availability = adapter_->availability();
  const RadarObjectsAdapterParams & params = adapter_->params();

  if (!availability.position_z()) {
    RCLCPP_WARN_ONCE(
      get_logger(),
      "The field position_z is not available in the radar info message. Defaulting to %f.",
      params.default_position_z);
  }

  if (!availability.velocity_z()) {
    RCLCPP_WARN_ONCE(
      get_logger(),
      "The field velocity_z is not available in the radar info message. Defaulting to %f.",
      params.default_velocity_z);
  }

  if (!availability.acceleration_z()) {
    RCLCPP_WARN_ONCE(
      get_logger(),
      "The field acceleration_z is not available in the radar info message. Defaulting to %f.",
      params.default_acceleration_z);
  }

  if (!availability.size_x()) {
    RCLCPP_WARN_ONCE(
      get_logger(),
      "The field size_x is not available in the radar info message. Defaulting to %f.",
      params.default_size_x);
  }

  if (!availability.size_y()) {
    RCLCPP_WARN_ONCE(
      get_logger(),
      "The field size_y is not available in the radar info message. Defaulting to %f.",
      params.default_size_y);
  }

  if (!availability.size_z()) {
    RCLCPP_WARN_ONCE(
      get_logger(),
      "The field size_z is not available in the radar info message. Defaulting to %f.",
      params.default_size_z);
  }
}

}  // namespace autoware::radar_objects_adapter

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::radar_objects_adapter::RadarObjectsAdapterNode)
