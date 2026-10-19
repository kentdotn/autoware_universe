// Copyright 2020 Tier IV, Inc.
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

#include "roi_visualizer.hpp"

#include "shape_draw.hpp"

#include <opencv2/imgproc.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>  // for ROS 2 Jazzy or newer
#else
#include <cv_bridge/cv_bridge.h>  // for ROS 2 Humble or older
#endif
#include <opencv2/imgproc/imgproc_c.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace autoware::traffic_light
{
namespace
{
// The palette, named after the signal colors; the message calls yellow AMBER.
const cv::Scalar red_color{230, 115, 115};    // #E67373
const cv::Scalar yellow_color{242, 191, 36};  // #F2BF24
const cv::Scalar green_color{153, 255, 178};  // #99FFB2
// Every other case: an unrecognized circle color, no circle in the signal, or no signal at all.
const cv::Scalar unknown_color{250, 250, 250};  // #FAFAFA

/**
 * @brief The color the frames of one traffic light are drawn in.
 *
 * Only a circle carries a color the palette names. An arrow, a color code the palette does not
 * name and a signal with no circle in it all come out in `unknown_color`. A traffic light shows
 * one circle at a time; should a signal report several, the last of them decides.
 */
cv::Scalar frame_color(
  const std::vector<tier4_perception_msgs::msg::TrafficLightElement> & elements)
{
  using tier4_perception_msgs::msg::TrafficLightElement;
  static const std::map<uint8_t, cv::Scalar> palette{
    {TrafficLightElement::RED, red_color},
    {TrafficLightElement::AMBER, yellow_color},
    {TrafficLightElement::GREEN, green_color},
  };

  cv::Scalar color = unknown_color;
  for (const auto & element : elements) {
    if (element.shape != TrafficLightElement::CIRCLE) {
      continue;
    }
    const auto found = palette.find(element.color);
    color = found == palette.end() ? unknown_color : found->second;
  }
  return color;
}

/// The shapes the label box holds, one per lamp, in the order they were reported.
std::vector<uint8_t> shape_codes(
  const std::vector<tier4_perception_msgs::msg::TrafficLightElement> & elements)
{
  std::vector<uint8_t> shapes;
  shapes.reserve(elements.size());
  for (const auto & element : elements) {
    shapes.push_back(element.shape);
  }
  return shapes;
}

/// The confidence the label box shows. All the lamps of one signal carry the same one.
float confidence_of(const std::vector<tier4_perception_msgs::msg::TrafficLightElement> & elements)
{
  return elements.empty() ? 0.0f : static_cast<float>(elements.back().confidence);
}

/**
 * @brief The lamps reported for `id`, or nullptr if no signal was reported for it.
 *
 * Should an array carry the id more than once - which the detection chain is not expected to
 * produce - the first of them is the result, the choice get_roi_from_id makes for the ROIs.
 */
const std::vector<tier4_perception_msgs::msg::TrafficLightElement> * find_classification(
  int id, const tier4_perception_msgs::msg::TrafficLightArray & traffic_signals)
{
  const auto found = std::find_if(
    traffic_signals.signals.begin(), traffic_signals.signals.end(),
    [id](const auto & traffic_signal) { return id == traffic_signal.traffic_light_id; });
  return found == traffic_signals.signals.end() ? nullptr : &found->elements;
}

bool get_roi_from_id(
  int id, const tier4_perception_msgs::msg::TrafficLightRoiArray & rois,
  tier4_perception_msgs::msg::TrafficLightRoi & correspond_roi)
{
  for (const auto roi : rois.rois) {
    if (roi.traffic_light_id == id) {
      correspond_roi = roi;
      return true;
    }
  }
  return false;
}
/// Draws the ROI in `color` and writes its traffic light id next to it.
bool draw_roi_with_id(
  cv::Mat & image, const tier4_perception_msgs::msg::TrafficLightRoi & tl_roi,
  const cv::Scalar & color)
{
  cv::rectangle(
    image, cv::Point(tl_roi.roi.x_offset, tl_roi.roi.y_offset),
    cv::Point(tl_roi.roi.x_offset + tl_roi.roi.width, tl_roi.roi.y_offset + tl_roi.roi.height),
    color, 3);
  cv::putText(
    image, std::to_string(tl_roi.traffic_light_id),
    cv::Point(tl_roi.roi.x_offset, tl_roi.roi.y_offset), cv::FONT_HERSHEY_COMPLEX, 1.0, color, 1,
    CV_AA);
  return true;
}
}  // namespace

TrafficLightRoiVisualizer::TrafficLightRoiVisualizer(std::string shape_image_dir)
: shape_image_dir_(std::move(shape_image_dir))
{
}

bool TrafficLightRoiVisualizer::draw_roi_with_label(
  cv::Mat & image, const tier4_perception_msgs::msg::TrafficLightRoi & tl_roi,
  const std::vector<tier4_perception_msgs::msg::TrafficLightElement> & elements) const
{
  const auto color = frame_color(elements);

  cv::rectangle(
    image, cv::Point(tl_roi.roi.x_offset, tl_roi.roi.y_offset),
    cv::Point(tl_roi.roi.x_offset + tl_roi.roi.width, tl_roi.roi.y_offset + tl_roi.roi.height),
    color, 2);

  constexpr int shape_img_size = 16;
  const auto position = cv::Point(tl_roi.roi.x_offset, tl_roi.roi.y_offset);

  visualization::draw_traffic_light_shape(
    image, shape_image_dir_, shape_codes(elements), shape_img_size, position, color,
    confidence_of(elements));

  return true;
}

sensor_msgs::msg::Image::SharedPtr TrafficLightRoiVisualizer::visualize(
  const sensor_msgs::msg::Image & image,
  const tier4_perception_msgs::msg::TrafficLightRoiArray & rois,
  const tier4_perception_msgs::msg::TrafficLightArray & traffic_signals) const
{
  // Convert to RGB8 from whatever the camera sent, since the drawing below only handles RGB8.
  const auto cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);

  for (auto tl_roi : rois.rois) {
    const auto * elements = find_classification(tl_roi.traffic_light_id, traffic_signals);

    if (elements == nullptr) {
      // does not have classification result
      draw_roi_with_id(cv_ptr->image, tl_roi, unknown_color);
    } else {
      // has classification result
      draw_roi_with_label(cv_ptr->image, tl_roi, *elements);
    }
  }

  return cv_ptr->toImageMsg();
}

sensor_msgs::msg::Image::SharedPtr TrafficLightRoiVisualizer::visualize_with_rough_rois(
  const sensor_msgs::msg::Image & image,
  const tier4_perception_msgs::msg::TrafficLightRoiArray & rois,
  const tier4_perception_msgs::msg::TrafficLightRoiArray & rough_rois,
  const tier4_perception_msgs::msg::TrafficLightArray & traffic_signals) const
{
  // Convert to RGB8 from whatever the camera sent, since the drawing below only handles RGB8.
  const auto cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);

  for (auto tl_rough_roi : rough_rois.rois) {
    // note: a signal will still be output even if it is undetected
    // Its position and size will be set as 0 and the color will be set as unknown
    // So a rough roi will always have correspond roi a correspond traffic signal
    const auto * elements = find_classification(tl_rough_roi.traffic_light_id, traffic_signals);
    const auto signal_color = elements == nullptr ? unknown_color : frame_color(*elements);
    tier4_perception_msgs::msg::TrafficLightRoi tl_roi;
    bool has_correspond_roi = get_roi_from_id(tl_rough_roi.traffic_light_id, rois, tl_roi);

    if (has_correspond_roi) {
      draw_roi_with_id(cv_ptr->image, tl_rough_roi, signal_color);

      if (elements != nullptr) {
        // has fine detection and classification results
        draw_roi_with_label(cv_ptr->image, tl_roi, *elements);
      } else {
        // has fine detection result and does not have classification result
        draw_roi_with_id(cv_ptr->image, tl_roi, unknown_color);
      }
    } else {
      // these cases are not expected, but could happen on some unpredictable conditions

      if (elements != nullptr) {
        // does not have fine detection result and has classification result: the label goes on
        // the rough ROI, so it takes the place of the id rather than being painted over it
        draw_roi_with_label(cv_ptr->image, tl_rough_roi, *elements);
      } else {
        draw_roi_with_id(cv_ptr->image, tl_rough_roi, signal_color);
      }
    }
  }

  return cv_ptr->toImageMsg();
}

}  // namespace autoware::traffic_light
