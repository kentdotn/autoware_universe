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
#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>  // for ROS 2 Jazzy or newer
#else
#include <cv_bridge/cv_bridge.h>  // for ROS 2 Humble or older
#endif
#include <opencv2/imgproc/imgproc_c.h>

#include <cstdint>
#include <string>
#include <vector>

namespace autoware::traffic_light::visualization
{
/**
 * @brief A struct of parameters to load shape image.
 */
struct ShapeImgParam
{
  std::string filename;  //!< Filename of shape image.
  bool h_flip;           //!< Whether to flip horizontally
  bool v_flip;           //!< Whether to flip vertically
};

/**
 * @brief Draw traffic light shapes on the camera view image.
 * @param image Camera view image.
 * @param image_dir Directory holding the shape images, passed in so that this file needs no
 * package lookup of its own.
 * @param params Shape parameters to load shape image.
 * @param size Shape image size to resize.
 * @param position Top-left position of a ROI.
 * @param color Rectangle color.
 * @param probability Classification probability.
 */
void draw_shape(
  cv::Mat & image, const std::string & image_dir, const std::vector<ShapeImgParam> & params,
  int size, const cv::Point & position, const cv::Scalar & color, float probability);

/**
 * @brief Load shape images and concatenate them.
 * @param image_dir Directory holding the shape images.
 * @param params Parameters for each shape image.
 * @param size Image size to resize.
 * @param scale_factor Scale factor to resize.
 * @return If no parameter is specified returns empty Mat, otherwise returns horizontally
 * concatenated image.
 */
cv::Mat load_shape_image(
  const std::string & image_dir, const std::vector<ShapeImgParam> & params, int size,
  double scale_factor = 0.3);

/**
 * @brief Draw traffic light shapes on the camera view image.
 * @param image Camera view image.
 * @param image_dir Directory holding the shape images.
 * @param shapes Shape codes, as TrafficLightElement defines them. A code with no image of its
 * own is drawn as the unknown shape.
 * @param size Shape image size to resize.
 * @param position Top-left position of a ROI.
 * @param color Color of traffic light.
 * @param probability Classification probability.
 */
void draw_traffic_light_shape(
  cv::Mat & image, const std::string & image_dir, const std::vector<uint8_t> & shapes, int size,
  const cv::Point & position, const cv::Scalar & color, float probability);

}  // namespace autoware::traffic_light::visualization
