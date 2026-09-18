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

#ifndef RADAR_OBJECTS_ADAPTER_NODE_HPP_
#define RADAR_OBJECTS_ADAPTER_NODE_HPP_

#include "radar_objects_adapter.hpp"

#include <autoware/agnocast_wrapper/node.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_sensing_msgs/msg/radar_info.hpp>
#include <autoware_sensing_msgs/msg/radar_objects.hpp>

#include <optional>

namespace autoware::radar_objects_adapter
{
// The ROS side of the adapter: topics, parameters and logging. The conversion itself is
// RadarObjectsAdapter (radar_objects_adapter.hpp).
class RadarObjectsAdapterNode : public autoware::agnocast_wrapper::Node
{
public:
  explicit RadarObjectsAdapterNode(const rclcpp::NodeOptions & options);

private:
  void objects_callback(const autoware_sensing_msgs::msg::RadarObjects & objects_msg);
  void radar_info_callback(const autoware_sensing_msgs::msg::RadarInfo & radar_info_msg);

  AUTOWARE_SUBSCRIPTION_PTR(autoware_sensing_msgs::msg::RadarObjects) radar_objects_sub_;
  AUTOWARE_SUBSCRIPTION_PTR(autoware_sensing_msgs::msg::RadarInfo) radar_info_sub_;
  AUTOWARE_PUBLISHER_PTR(autoware_perception_msgs::msg::DetectedObjects) detections_pub_;
  AUTOWARE_PUBLISHER_PTR(autoware_perception_msgs::msg::TrackedObjects) tracks_pub_;

  std::optional<RadarObjectsAdapter> adapter_;
};
}  // namespace autoware::radar_objects_adapter

#endif  // RADAR_OBJECTS_ADAPTER_NODE_HPP_
