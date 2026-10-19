// Copyright 2024 The Autoware Contributors
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
#include "shape_draw.hpp"

#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <tier4_perception_msgs/msg/traffic_light_element.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace autoware::traffic_light::visualization
{
void draw_shape(
  cv::Mat & image, const std::string & image_dir, const std::vector<ShapeImgParam> & params,
  int size, const cv::Point & position, const cv::Scalar & color, float probability)
{
  // skip if the roi position is set as (0,0), which means it is undetected
  if (position.x == 0 && position.y == 0) {
    return;
  }
  // load concatenated shape image
  const auto shape_img = load_shape_image(image_dir, params, size);

  // Calculate the width of the text
  std::string prob_str = std::to_string(static_cast<int>(round(probability * 100))) + "%";

  int baseline = 0;
  cv::Size text_size = cv::getTextSize(prob_str, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseline);

  const int fill_rect_w = shape_img.cols + text_size.width + 20;
  const int fill_rect_h = std::max(shape_img.rows, text_size.height) + 10;

  const cv::Point rect_position(position.x, position.y - fill_rect_h);
  if (
    rect_position.x < 0 || rect_position.y < 0 || rect_position.x + fill_rect_w > image.cols ||
    position.y > image.rows) {
    // TODO(KhalilSelyan): This error message may flood the terminal logs, so commented out
    // temporarily. Need to consider a better way.

    std::cerr << "Adjusted position is out of image bounds." << std::endl;
    return;
  }
  cv::Rect rectangle(rect_position.x, rect_position.y, fill_rect_w, fill_rect_h);
  cv::rectangle(image, rectangle, color, -1);

  // Position the probability text right next to the shape. (Text origin: bottom-left)
  const int prob_x_offset = shape_img.cols + 15;
  const int prob_y_offset = fill_rect_h / 4;
  cv::putText(
    image, prob_str, cv::Point(position.x + prob_x_offset, position.y - prob_y_offset),
    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

  if (!shape_img.empty()) {
    // Create ROI on the destination image
    const int shape_y_offset = fill_rect_h / 4;
    auto shape_roi = image(
      cv::Rect(
        rect_position.x + 5, rect_position.y + shape_y_offset, shape_img.cols, shape_img.rows));

    // Overlay the image onto the main image
    for (int y = 0; y < shape_img.rows; ++y) {
      for (int x = 0; x < shape_img.cols; ++x) {
        const auto & pixel = shape_img.at<cv::Vec4b>(y, x);
        if (pixel[3] != 0) {  // Only non-transparent pixels
          shape_roi.at<cv::Vec3b>(y, x) = cv::Vec3b(pixel[0], pixel[1], pixel[2]);
        }
      }
    }
  }
}

cv::Mat load_shape_image(
  const std::string & image_dir, const std::vector<ShapeImgParam> & params, int size,
  double scale_factor)
{
  if (params.empty()) {
    return {};
  }

  std::vector<cv::Mat> src_img;
  for (const auto & param : params) {
    auto filepath = image_dir + param.filename;
    auto img = cv::imread(filepath, cv::IMREAD_UNCHANGED);

    cv::resize(img, img, cv::Size(size, size), scale_factor, scale_factor, cv::INTER_AREA);

    if (param.h_flip) {
      cv::flip(img, img, 1);
    }
    if (param.v_flip) {
      cv::flip(img, img, 0);
    }
    src_img.emplace_back(img);
  }

  cv::Mat dst;
  cv::hconcat(src_img, dst);  // cspell:ignore hconcat

  return dst;
}

void draw_traffic_light_shape(
  cv::Mat & image, const std::string & image_dir, const std::vector<uint8_t> & shapes, int size,
  const cv::Point & position, const cv::Scalar & color, float probability)
{
  using tier4_perception_msgs::msg::TrafficLightElement;

  // One image per shape, some of them the same file flipped. A code the table does not name is
  // drawn as the unknown shape, so that something is shown rather than nothing.
  static const std::map<uint8_t, ShapeImgParam> shape_to_param{
    {TrafficLightElement::CIRCLE, {"circle.png", false, false}},
    {TrafficLightElement::LEFT_ARROW, {"left_arrow.png", false, false}},
    {TrafficLightElement::RIGHT_ARROW, {"left_arrow.png", true, false}},
    {TrafficLightElement::UP_ARROW, {"straight_arrow.png", false, false}},
    {TrafficLightElement::DOWN_ARROW, {"straight_arrow.png", false, true}},
    {TrafficLightElement::UP_LEFT_ARROW, {"down_left_arrow.png", false, true}},
    {TrafficLightElement::UP_RIGHT_ARROW, {"down_left_arrow.png", true, true}},
    {TrafficLightElement::DOWN_LEFT_ARROW, {"down_left_arrow.png", false, false}},
    {TrafficLightElement::DOWN_RIGHT_ARROW, {"down_left_arrow.png", true, false}},
    {TrafficLightElement::CROSS, {"cross.png", false, false}},
    {TrafficLightElement::UNKNOWN, {"unknown.png", false, false}},
  };

  std::vector<ShapeImgParam> params;
  params.reserve(shapes.size());
  for (const auto & shape : shapes) {
    const auto found = shape_to_param.find(shape);
    params.emplace_back(
      found == shape_to_param.end() ? shape_to_param.at(TrafficLightElement::UNKNOWN)
                                    : found->second);
  }

  draw_shape(image, image_dir, params, size, position, color, probability);
}
}  // namespace autoware::traffic_light::visualization
