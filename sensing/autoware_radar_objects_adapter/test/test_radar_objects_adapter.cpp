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
//
// Unit tests of RadarObjectsAdapter, through what the node uses of it: the constructor, the
// classification remap built from the parameter names, update_radar_info() and convert(). They run
// without a ROS context; the node layer is covered by the node test. Nothing here reads the
// adapter's internals, so the implementation can be restructured under these tests. Expected values
// are derived from the definition of each step (a 2x2 rotation, a quaternion about z), not from the
// implementation.

#include "radar_objects_adapter.hpp"

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_sensing_msgs/msg/radar_classification.hpp>
#include <autoware_sensing_msgs/msg/radar_info.hpp>
#include <autoware_sensing_msgs/msg/radar_objects.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using autoware::radar_objects_adapter::ClassificationRemap;
using autoware::radar_objects_adapter::make_classification_remap_from_string_pair;
using autoware::radar_objects_adapter::RadarObjectsAdapter;
using autoware::radar_objects_adapter::RadarObjectsAdapterParams;
using autoware_perception_msgs::msg::DetectedObject;
using autoware_perception_msgs::msg::DetectedObjectKinematics;
using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::Shape;
using autoware_perception_msgs::msg::TrackedObject;
using autoware_perception_msgs::msg::TrackedObjectKinematics;
using autoware_sensing_msgs::msg::RadarClassification;
using autoware_sensing_msgs::msg::RadarFieldInfo;
using autoware_sensing_msgs::msg::RadarInfo;
using autoware_sensing_msgs::msg::RadarObject;
using autoware_sensing_msgs::msg::RadarObjects;
using Outcome = RadarObjectsAdapter::Outcome;
using RadarInfoResult = RadarObjectsAdapter::RadarInfoResult;
using Result = RadarObjectsAdapter::Result;
using NameMap = std::map<std::string, std::string>;

constexpr double tolerance = 1e-6;

// Parameter values that no radar object below carries, so that a field filled from a parameter
// can be told apart from one copied out of the object.
RadarObjectsAdapterParams make_params()
{
  RadarObjectsAdapterParams params;
  params.default_position_z = 0.25;
  params.default_velocity_z = 0.5;
  params.default_acceleration_z = 0.75;
  params.default_size_x = 5.0;
  params.default_size_y = 2.0;
  params.default_size_z = 1.5;
  return params;
}

// A remap with an identity entry for every radar label the parameters can name, so that the
// tests about kinematics are not disturbed by the classification.
ClassificationRemap identity_remap()
{
  ClassificationRemap remap;
  for (std::uint8_t label :
       {RadarClassification::UNKNOWN, RadarClassification::CAR, RadarClassification::TRUCK,
        RadarClassification::MOTORCYCLE, RadarClassification::BICYCLE,
        RadarClassification::PEDESTRIAN, RadarClassification::ANIMAL,
        RadarClassification::HAZARD}) {
    remap[label] = label;
  }
  return remap;
}

// The input topic the node would be subscribed to; its hash is part of every tracked object's id.
constexpr char topic_name[] = "/sensing/radar/front/objects_raw";

// The 8 bytes of the topic name's std::hash, least significant byte first.
std::array<std::uint8_t, 8> topic_hash_bytes(const std::string & topic)
{
  const std::size_t hash = std::hash<std::string>{}(topic);
  std::array<std::uint8_t, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((hash >> (i * 8)) & 0xFF);
  }
  return bytes;
}

RadarInfo make_radar_info(const std::vector<std::string> & field_names)
{
  RadarInfo info;
  for (const auto & name : field_names) {
    RadarFieldInfo field;
    field.field_name.data = name;
    info.object_fields_info.push_back(field);
  }
  return info;
}

const std::vector<std::string> required_fields = {
  "existence_probability", "position_x",     "position_y", "velocity_x", "velocity_y",
  "acceleration_x",        "acceleration_y", "orientation"};
const std::vector<std::string> optional_fields = {
  "position_z", "velocity_z", "acceleration_z",  "size_x",
  "size_y",     "size_z",     "orientation_std", "orientation_rate_std"};

std::vector<std::string> with(
  std::vector<std::string> fields, const std::vector<std::string> & added)
{
  fields.insert(fields.end(), added.begin(), added.end());
  return fields;
}

std::vector<std::string> without(std::vector<std::string> fields, const std::string & removed)
{
  fields.erase(std::find(fields.begin(), fields.end(), removed));
  return fields;
}

// The six optional fields that have a parameter, with the value make_params() gives each, in the
// order update_radar_info() reports them. orientation_std and orientation_rate_std have no
// parameter and are never reported.
using DefaultedField = std::pair<std::string, double>;
const std::vector<DefaultedField> all_defaulted = {{"position_z", 0.25},     {"velocity_z", 0.5},
                                                   {"acceleration_z", 0.75}, {"size_x", 5.0},
                                                   {"size_y", 2.0},          {"size_z", 1.5}};

std::vector<DefaultedField> all_defaulted_except(const std::string & declared)
{
  std::vector<DefaultedField> fields;
  for (const auto & field : all_defaulted) {
    if (field.first != declared) {
      fields.push_back(field);
    }
  }
  return fields;
}

// An adapter that has seen no radar info.
RadarObjectsAdapter make_fresh_adapter()
{
  return RadarObjectsAdapter(make_params(), identity_remap(), topic_name);
}

RadarClassification make_classification(std::uint8_t label, float probability)
{
  RadarClassification classification;
  classification.label = label;
  classification.probability = probability;
  return classification;
}

// A radar object with every field set to a distinct, recognizable value.
RadarObject make_radar_object()
{
  RadarObject object;
  object.object_id = 0x04030201u;
  object.age = 12;
  object.measurement_status = RadarObject::MEASUREMENT_STATUS_MEASURED;
  object.movement_status = RadarObject::MOVEMENT_STATUS_DYNAMIC;
  object.position.x = 10.0;
  object.position.y = -2.0;
  object.position.z = 0.8;
  object.velocity.x = 3.0;
  object.velocity.y = 0.5;
  object.velocity.z = 0.1;
  object.acceleration.x = 0.2;
  object.acceleration.y = -0.1;
  object.acceleration.z = 0.05;
  object.size.x = 4.5;
  object.size.y = 1.8;
  object.size.z = 1.4;
  // Upper triangles in the order XX, XY, XZ, YY, YZ, ZZ.
  object.position_covariance = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  object.velocity_covariance = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  object.acceleration_covariance = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f};
  object.size_covariance = {0.7f, 0.8f, 0.9f, 1.1f, 1.2f, 1.3f};
  object.orientation = 0.3f;
  object.orientation_std = 0.1f;
  object.orientation_rate = 0.05f;
  object.orientation_rate_std = 0.02f;
  object.existence_probability = 0.9f;
  object.classifications = {make_classification(RadarClassification::CAR, 0.8f)};
  return object;
}

RadarObject facing(RadarObject object, double yaw)
{
  object.orientation = static_cast<float>(yaw);
  return object;
}

RadarObjects make_radar_objects(const std::vector<RadarObject> & objects)
{
  RadarObjects msg;
  msg.header.stamp.sec = 1700000000;
  msg.header.stamp.nanosec = 100;
  msg.header.frame_id = "base_link";
  msg.objects = objects;
  return msg;
}

// An adapter that has seen a radar info declaring the required fields plus `optional`.
RadarObjectsAdapter make_adapter(const std::vector<std::string> & optional = optional_fields)
{
  RadarObjectsAdapter adapter(make_params(), identity_remap(), topic_name);
  adapter.update_radar_info(make_radar_info(with(required_fields, optional)));
  return adapter;
}

// Converts one object and returns what came out on both outputs.
struct Converted
{
  DetectedObject detected;
  TrackedObject tracked;
};

Converted convert_one(const RadarObjectsAdapter & adapter, const RadarObject & object)
{
  const Result result = adapter.convert(make_radar_objects({object}));
  EXPECT_EQ(result.outcome, Outcome::Converted);
  return {result.detections.value().objects.at(0), result.tracks.value().objects.at(0)};
}

// Indices into the 6x6 row-major covariance (x, y, z, roll, pitch, yaw) of a perception message.
constexpr size_t cov_x_x = 0;
constexpr size_t cov_x_y = 1;
constexpr size_t cov_y_x = 6;
constexpr size_t cov_y_y = 7;
constexpr size_t cov_yaw_yaw = 35;

// A vector in the radar frame, expressed in the frame of an object facing `yaw`: the rotation by
// -yaw, written out.
std::pair<double, double> into_object_frame(double x, double y, double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  return {c * x + s * y, -s * x + c * y};
}

// The 2x2 covariance [[xx, xy], [xy, yy]] of a vector in the radar frame, expressed in the frame
// of an object facing `yaw`: R Sigma R^T with R the rotation by -yaw, written out.
struct Covariance2x2
{
  double xx;
  double xy;
  double yy;
};

Covariance2x2 into_object_frame(const Covariance2x2 & cov, double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  // R = [[c, s], [-s, c]]; R Sigma = [[c xx + s xy, c xy + s yy], [-s xx + c xy, -s xy + c yy]]
  const double r00 = c * cov.xx + s * cov.xy;
  const double r01 = c * cov.xy + s * cov.yy;
  const double r10 = -s * cov.xx + c * cov.xy;
  const double r11 = -s * cov.xy + c * cov.yy;
  // (R Sigma) R^T
  return {r00 * c + r01 * s, r00 * (-s) + r01 * c, r10 * (-s) + r11 * c};
}

// The yaw angles the rotation cases are run at: the trivial one, the one that swaps the axes, a
// general one, and one past a half turn.
const std::vector<double> yaws = {0.0, M_PI / 2.0, 0.3, -2.5};

// Every entry of a 6x6 covariance other than the ones named is zero.
bool only_these_entries_set(
  const std::array<double, 36> & covariance, const std::vector<size_t> & set_indices)
{
  for (size_t i = 0; i < covariance.size(); ++i) {
    const bool is_set = std::find(set_indices.begin(), set_indices.end(), i) != set_indices.end();
    if (!is_set && covariance[i] != 0.0) {
      return false;
    }
  }
  return true;
}

using LabeledProbability = std::pair<std::uint8_t, float>;

std::vector<LabeledProbability> labeled_probabilities(
  const std::vector<ObjectClassification> & classifications)
{
  std::vector<LabeledProbability> result;
  for (const auto & classification : classifications) {
    result.emplace_back(classification.label, classification.probability);
  }
  return result;
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// The radar info: what update_radar_info() reports about a message, and what it makes convert()
// do afterwards (copy a field or fill it from a parameter, which the Fields_ cases below read).
// ---------------------------------------------------------------------------------------------

// A radar info that declares the required fields and nothing else is valid, and every optional
// field that has a parameter is reported as filled from it, with the parameter's value.
TEST(RadarObjectsAdapter, RadarInfo_AllRequiredFields_ValidAndAllDefaulted)
{
  RadarObjectsAdapter adapter = make_fresh_adapter();

  const RadarInfoResult result = adapter.update_radar_info(make_radar_info(required_fields));

  EXPECT_TRUE(result.valid());
  EXPECT_TRUE(result.missing_required_fields.empty());
  EXPECT_EQ(result.defaulted_fields, all_defaulted);
}

// Each of the eight required fields is required on its own: leaving out any one of them keeps
// the radar info invalid, the result names exactly that field, and nothing is defaulted.
TEST(RadarObjectsAdapter, RadarInfo_OneRequiredFieldMissing_InvalidAndNamed)
{
  for (const auto & missing : required_fields) {
    SCOPED_TRACE(missing);
    RadarObjectsAdapter adapter = make_fresh_adapter();

    const RadarInfoResult result =
      adapter.update_radar_info(make_radar_info(without(required_fields, missing)));

    EXPECT_FALSE(result.valid());
    EXPECT_EQ(result.missing_required_fields, std::vector<std::string>{missing});
    EXPECT_TRUE(result.defaulted_fields.empty());
  }
}

// When several required fields are missing, all of them are named, in the order they are
// required, whatever order the radar info declared the others in.
TEST(RadarObjectsAdapter, RadarInfo_SeveralRequiredFieldsMissing_AllNamedInRequiredOrder)
{
  RadarObjectsAdapter adapter = make_fresh_adapter();

  const RadarInfoResult result =
    adapter.update_radar_info(make_radar_info({"velocity_y", "position_x", "orientation"}));

  EXPECT_FALSE(result.valid());
  const std::vector<std::string> expected = {
    "existence_probability", "position_y", "velocity_x", "acceleration_x", "acceleration_y"};
  EXPECT_EQ(result.missing_required_fields, expected);
}

// A radar info with no object fields at all names every required field.
TEST(RadarObjectsAdapter, RadarInfo_NoFields_AllRequiredFieldsNamed)
{
  RadarObjectsAdapter adapter = make_fresh_adapter();

  const RadarInfoResult result = adapter.update_radar_info(make_radar_info({}));

  EXPECT_FALSE(result.valid());
  EXPECT_EQ(result.missing_required_fields, required_fields);
}

// Each optional field, once declared, drops out of the defaulted list and only it does.
TEST(RadarObjectsAdapter, RadarInfo_OneOptionalField_OnlyItNotDefaulted)
{
  for (const auto & declared : optional_fields) {
    SCOPED_TRACE(declared);
    RadarObjectsAdapter adapter = make_fresh_adapter();

    const RadarInfoResult result =
      adapter.update_radar_info(make_radar_info(with(required_fields, {declared})));

    ASSERT_TRUE(result.valid());
    EXPECT_EQ(result.defaulted_fields, all_defaulted_except(declared));
  }
}

// With every optional field declared, nothing is defaulted.
TEST(RadarObjectsAdapter, RadarInfo_AllOptionalFields_NothingDefaulted)
{
  RadarObjectsAdapter adapter = make_fresh_adapter();

  const RadarInfoResult result =
    adapter.update_radar_info(make_radar_info(with(required_fields, optional_fields)));

  ASSERT_TRUE(result.valid());
  EXPECT_TRUE(result.defaulted_fields.empty());
}

// Fields the adapter does not know (the ones the radar in use declares on top of the ones above,
// and arbitrary names) are ignored: they neither help the required check nor count as optional.
TEST(RadarObjectsAdapter, RadarInfo_UnknownFields_Ignored)
{
  RadarObjectsAdapter adapter = make_fresh_adapter();

  const RadarInfoResult result = adapter.update_radar_info(make_radar_info(with(
    without(required_fields, "orientation"),
    {"object_id", "age", "measurement_status", "movement_status", "orientation_rate",
     "not_a_field"})));

  EXPECT_FALSE(result.valid());
  EXPECT_EQ(result.missing_required_fields, std::vector<std::string>{"orientation"});
}

// Deliberately not pinned here: what a second radar info does to the fields declared by the first
// (today they accumulate). The maintainers have answered that the first valid radar info should
// fix them; the change and its cases come with the refactoring.

// ---------------------------------------------------------------------------------------------
// The gate: convert() answers with NoValidRadarInfo and no messages until a radar info has
// declared every required field.
// ---------------------------------------------------------------------------------------------

TEST(RadarObjectsAdapter, Convert_NoRadarInfo_NoValidRadarInfo)
{
  const RadarObjectsAdapter adapter(make_params(), identity_remap(), topic_name);

  const Result result = adapter.convert(make_radar_objects({make_radar_object()}));

  EXPECT_EQ(result.outcome, Outcome::NoValidRadarInfo);
  EXPECT_FALSE(result.detections.has_value());
  EXPECT_FALSE(result.tracks.has_value());
}

TEST(RadarObjectsAdapter, Convert_InvalidRadarInfo_NoValidRadarInfo)
{
  RadarObjectsAdapter adapter(make_params(), identity_remap(), topic_name);
  const auto update = adapter.update_radar_info(make_radar_info({"position_x", "position_y"}));
  ASSERT_FALSE(update.valid());

  const Result result = adapter.convert(make_radar_objects({make_radar_object()}));

  EXPECT_EQ(result.outcome, Outcome::NoValidRadarInfo);
  EXPECT_FALSE(result.detections.has_value());
  EXPECT_FALSE(result.tracks.has_value());
}

// With a valid radar info, both messages come back, carrying the input header and one output
// object per input object, in order.
TEST(RadarObjectsAdapter, Convert_ValidRadarInfo_BothMessagesWithHeader)
{
  const RadarObjectsAdapter adapter = make_adapter();
  RadarObject first = make_radar_object();
  first.position.x = 1.0;
  RadarObject second = make_radar_object();
  second.position.x = 2.0;

  const Result result = adapter.convert(make_radar_objects({first, second}));

  ASSERT_EQ(result.outcome, Outcome::Converted);
  ASSERT_TRUE(result.detections.has_value());
  ASSERT_TRUE(result.tracks.has_value());
  EXPECT_EQ(result.detections->header.stamp.sec, 1700000000);
  EXPECT_EQ(result.detections->header.stamp.nanosec, 100u);
  EXPECT_EQ(result.detections->header.frame_id, "base_link");
  EXPECT_EQ(result.tracks->header, result.detections->header);
  ASSERT_EQ(result.detections->objects.size(), 2u);
  ASSERT_EQ(result.tracks->objects.size(), 2u);
  EXPECT_DOUBLE_EQ(
    result.detections->objects[0].kinematics.pose_with_covariance.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(
    result.detections->objects[1].kinematics.pose_with_covariance.pose.position.x, 2.0);
  EXPECT_DOUBLE_EQ(result.tracks->objects[0].kinematics.pose_with_covariance.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(result.tracks->objects[1].kinematics.pose_with_covariance.pose.position.x, 2.0);
}

TEST(RadarObjectsAdapter, Convert_NoObjects_EmptyMessages)
{
  const RadarObjectsAdapter adapter = make_adapter();

  const Result result = adapter.convert(make_radar_objects({}));

  ASSERT_EQ(result.outcome, Outcome::Converted);
  EXPECT_TRUE(result.detections->objects.empty());
  EXPECT_TRUE(result.tracks->objects.empty());
}

// ---------------------------------------------------------------------------------------------
// Fields copied or defaulted, depending on what the radar info declared.
// ---------------------------------------------------------------------------------------------

// Fields the radar always provides are copied as they are, on both outputs.
TEST(RadarObjectsAdapter, Fields_Required_Copied)
{
  const RadarObjectsAdapter adapter = make_adapter({});
  const RadarObject radar = make_radar_object();

  const auto [detected, tracked] = convert_one(adapter, radar);

  for (const auto * object :
       {&detected.kinematics.pose_with_covariance.pose.position,
        &tracked.kinematics.pose_with_covariance.pose.position}) {
    EXPECT_DOUBLE_EQ(object->x, radar.position.x);
    EXPECT_DOUBLE_EQ(object->y, radar.position.y);
  }
  EXPECT_FLOAT_EQ(detected.existence_probability, radar.existence_probability);
  EXPECT_FLOAT_EQ(tracked.existence_probability, radar.existence_probability);
  EXPECT_DOUBLE_EQ(
    detected.kinematics.twist_with_covariance.twist.angular.z,
    static_cast<double>(radar.orientation_rate));
  EXPECT_DOUBLE_EQ(
    tracked.kinematics.twist_with_covariance.twist.angular.z,
    static_cast<double>(radar.orientation_rate));
}

// Each optional value is copied from the object when its field is declared and taken from the
// parameter when it is not - and the choice is made per field.
TEST(RadarObjectsAdapter, Fields_Optional_CopiedWhenDeclaredDefaultedOtherwise)
{
  const RadarObject radar = make_radar_object();
  const RadarObjectsAdapterParams params = make_params();

  struct Case
  {
    std::string field;
    std::function<double(const DetectedObject &, const TrackedObject &)> read;
    double from_object;
    double from_params;
  };
  const std::vector<Case> cases = {
    {"position_z",
     [](const DetectedObject & d, const TrackedObject &) {
       return d.kinematics.pose_with_covariance.pose.position.z;
     },
     radar.position.z, params.default_position_z},
    {"velocity_z",
     [](const DetectedObject & d, const TrackedObject &) {
       return d.kinematics.twist_with_covariance.twist.linear.z;
     },
     radar.velocity.z, params.default_velocity_z},
    {"acceleration_z",
     [](const DetectedObject &, const TrackedObject & t) {
       return t.kinematics.acceleration_with_covariance.accel.linear.z;
     },
     radar.acceleration.z, params.default_acceleration_z},
    {"size_x", [](const DetectedObject & d, const TrackedObject &) { return d.shape.dimensions.x; },
     radar.size.x, params.default_size_x},
    {"size_y", [](const DetectedObject & d, const TrackedObject &) { return d.shape.dimensions.y; },
     radar.size.y, params.default_size_y},
    {"size_z", [](const DetectedObject & d, const TrackedObject &) { return d.shape.dimensions.z; },
     radar.size.z, params.default_size_z},
  };

  for (const auto & c : cases) {
    SCOPED_TRACE(c.field);
    {
      const auto [detected, tracked] = convert_one(make_adapter({c.field}), radar);
      EXPECT_DOUBLE_EQ(c.read(detected, tracked), c.from_object);
    }
    {
      const auto [detected, tracked] = convert_one(make_adapter({}), radar);
      EXPECT_DOUBLE_EQ(c.read(detected, tracked), c.from_params);
    }
  }
}

// The shape is always a bounding box; the tracked object gets the same dimensions as the
// detected one.
TEST(RadarObjectsAdapter, Fields_Shape_BoundingBoxOnBothOutputs)
{
  const auto [detected, tracked] = convert_one(make_adapter(), make_radar_object());

  EXPECT_EQ(detected.shape.type, Shape::BOUNDING_BOX);
  EXPECT_EQ(tracked.shape, detected.shape);
}

// The yaw variance is the square of the radar's orientation standard deviation when that field
// is declared, and stays zero when it is not; the same for the yaw rate.
TEST(RadarObjectsAdapter, Fields_YawVariances_SquaredWhenDeclaredZeroOtherwise)
{
  const RadarObject radar = make_radar_object();
  const double yaw_std = radar.orientation_std;
  const double yaw_rate_std = radar.orientation_rate_std;

  {
    const auto [detected, tracked] =
      convert_one(make_adapter({"orientation_std", "orientation_rate_std"}), radar);
    EXPECT_NEAR(
      detected.kinematics.pose_with_covariance.covariance[cov_yaw_yaw], yaw_std * yaw_std,
      tolerance);
    EXPECT_NEAR(
      detected.kinematics.twist_with_covariance.covariance[cov_yaw_yaw],
      yaw_rate_std * yaw_rate_std, tolerance);
  }
  {
    const auto [detected, tracked] = convert_one(make_adapter({}), radar);
    EXPECT_DOUBLE_EQ(detected.kinematics.pose_with_covariance.covariance[cov_yaw_yaw], 0.0);
    EXPECT_DOUBLE_EQ(detected.kinematics.twist_with_covariance.covariance[cov_yaw_yaw], 0.0);
  }
}

// ---------------------------------------------------------------------------------------------
// Orientation and the rotation into the object's frame.
// ---------------------------------------------------------------------------------------------

// The radar's yaw becomes a quaternion about z: (0, 0, sin(yaw/2), cos(yaw/2)).
TEST(RadarObjectsAdapter, Rotation_Orientation_QuaternionAboutZ)
{
  const RadarObjectsAdapter adapter = make_adapter();
  for (const double yaw : yaws) {
    SCOPED_TRACE(yaw);

    const auto [detected, tracked] = convert_one(adapter, facing(make_radar_object(), yaw));

    for (const auto * q :
         {&detected.kinematics.pose_with_covariance.pose.orientation,
          &tracked.kinematics.pose_with_covariance.pose.orientation}) {
      EXPECT_NEAR(q->x, 0.0, tolerance);
      EXPECT_NEAR(q->y, 0.0, tolerance);
      EXPECT_NEAR(q->z, std::sin(yaw / 2.0), tolerance);
      EXPECT_NEAR(q->w, std::cos(yaw / 2.0), tolerance);
    }
  }
}

// The velocity and the acceleration are rotated from the radar's frame into the object's frame
// by the object's yaw; the z components are not touched.
TEST(RadarObjectsAdapter, Rotation_VelocityAndAcceleration_IntoObjectFrame)
{
  const RadarObjectsAdapter adapter = make_adapter();
  for (const double yaw : yaws) {
    SCOPED_TRACE(yaw);
    const RadarObject radar = facing(make_radar_object(), yaw);

    const auto [detected, tracked] = convert_one(adapter, radar);

    const auto [vx, vy] = into_object_frame(radar.velocity.x, radar.velocity.y, yaw);
    for (const auto * twist :
         {&detected.kinematics.twist_with_covariance.twist,
          &tracked.kinematics.twist_with_covariance.twist}) {
      EXPECT_NEAR(twist->linear.x, vx, tolerance);
      EXPECT_NEAR(twist->linear.y, vy, tolerance);
      EXPECT_DOUBLE_EQ(twist->linear.z, radar.velocity.z);
    }

    const auto [ax, ay] = into_object_frame(radar.acceleration.x, radar.acceleration.y, yaw);
    const auto & accel = tracked.kinematics.acceleration_with_covariance.accel;
    EXPECT_NEAR(accel.linear.x, ax, tolerance);
    EXPECT_NEAR(accel.linear.y, ay, tolerance);
    EXPECT_DOUBLE_EQ(accel.linear.z, radar.acceleration.z);
  }
}

// The x/y block of the velocity and acceleration covariances is rotated with the vectors
// (R Sigma R^T) and stays symmetric; the z entries of the radar covariance are dropped.
TEST(RadarObjectsAdapter, Rotation_VelocityAndAccelerationCovariance_RotatedWithVectors)
{
  const RadarObjectsAdapter adapter = make_adapter({});  // no yaw rate variance in the way
  for (const double yaw : yaws) {
    SCOPED_TRACE(yaw);
    RadarObject radar = facing(make_radar_object(), yaw);
    radar.velocity_covariance = {1.0f, 0.5f, 9.0f, 4.0f, 9.0f, 9.0f};
    radar.acceleration_covariance = {2.0f, -0.3f, 9.0f, 0.5f, 9.0f, 9.0f};

    const auto [detected, tracked] = convert_one(adapter, radar);

    const Covariance2x2 velocity = into_object_frame({1.0, 0.5, 4.0}, yaw);
    for (const auto * cov :
         {&detected.kinematics.twist_with_covariance.covariance,
          &tracked.kinematics.twist_with_covariance.covariance}) {
      EXPECT_NEAR((*cov)[cov_x_x], velocity.xx, tolerance);
      EXPECT_NEAR((*cov)[cov_x_y], velocity.xy, tolerance);
      EXPECT_NEAR((*cov)[cov_y_x], velocity.xy, tolerance);
      EXPECT_NEAR((*cov)[cov_y_y], velocity.yy, tolerance);
      EXPECT_TRUE(only_these_entries_set(*cov, {cov_x_x, cov_x_y, cov_y_x, cov_y_y}));
    }

    const Covariance2x2 acceleration = into_object_frame({2.0, -0.3, 0.5}, yaw);
    const auto & cov = tracked.kinematics.acceleration_with_covariance.covariance;
    EXPECT_NEAR(cov[cov_x_x], acceleration.xx, tolerance);
    EXPECT_NEAR(cov[cov_x_y], acceleration.xy, tolerance);
    EXPECT_NEAR(cov[cov_y_x], acceleration.xy, tolerance);
    EXPECT_NEAR(cov[cov_y_y], acceleration.yy, tolerance);
    EXPECT_TRUE(only_these_entries_set(cov, {cov_x_x, cov_x_y, cov_y_x, cov_y_y}));
  }
}

// The position covariance is expressed in the radar's frame like the position itself, so its
// x/y block is copied without rotation, whatever the yaw; the z entries are dropped.
TEST(RadarObjectsAdapter, Rotation_PositionCovariance_NotRotated)
{
  const RadarObjectsAdapter adapter = make_adapter({});
  for (const double yaw : yaws) {
    SCOPED_TRACE(yaw);
    RadarObject radar = facing(make_radar_object(), yaw);
    radar.position_covariance = {1.0f, 2.0f, 9.0f, 4.0f, 9.0f, 9.0f};

    const auto [detected, tracked] = convert_one(adapter, radar);

    for (const auto * cov :
         {&detected.kinematics.pose_with_covariance.covariance,
          &tracked.kinematics.pose_with_covariance.covariance}) {
      EXPECT_NEAR((*cov)[cov_x_x], 1.0, tolerance);
      EXPECT_NEAR((*cov)[cov_x_y], 2.0, tolerance);
      EXPECT_NEAR((*cov)[cov_y_x], 2.0, tolerance);
      EXPECT_NEAR((*cov)[cov_y_y], 4.0, tolerance);
      EXPECT_TRUE(only_these_entries_set(*cov, {cov_x_x, cov_x_y, cov_y_x, cov_y_y}));
    }
  }
}

// INVALID_COV_VALUE (-1) marks an entry the radar cannot provide; it is read as zero, in every
// one of the three covariances, before any rotation.
TEST(RadarObjectsAdapter, Covariance_InvalidValue_ReadAsZero)
{
  const RadarObjectsAdapter adapter = make_adapter({});
  RadarObject radar = facing(make_radar_object(), 0.3);
  radar.position_covariance.fill(RadarObject::INVALID_COV_VALUE);
  radar.velocity_covariance.fill(RadarObject::INVALID_COV_VALUE);
  radar.acceleration_covariance.fill(RadarObject::INVALID_COV_VALUE);

  const auto [detected, tracked] = convert_one(adapter, radar);

  EXPECT_TRUE(only_these_entries_set(detected.kinematics.pose_with_covariance.covariance, {}));
  EXPECT_TRUE(only_these_entries_set(detected.kinematics.twist_with_covariance.covariance, {}));
  EXPECT_TRUE(
    only_these_entries_set(tracked.kinematics.acceleration_with_covariance.covariance, {}));
}

// ---------------------------------------------------------------------------------------------
// What is specific to each output.
// ---------------------------------------------------------------------------------------------

// The detected object declares everything it carries as present and the orientation as known,
// whether the flagged values came from the radar or from a parameter: the flags say what the
// message carries, not where a value came from.
TEST(RadarObjectsAdapter, Detected_KinematicsFlags_AllSetWithOnlyRequiredFields)
{
  const auto [detected, tracked] = convert_one(make_adapter({}), make_radar_object());

  EXPECT_TRUE(detected.kinematics.has_position_covariance);
  EXPECT_TRUE(detected.kinematics.has_twist);
  EXPECT_TRUE(detected.kinematics.has_twist_covariance);
  EXPECT_EQ(detected.kinematics.orientation_availability, DetectedObjectKinematics::AVAILABLE);
  EXPECT_EQ(tracked.kinematics.orientation_availability, TrackedObjectKinematics::AVAILABLE);
}

TEST(RadarObjectsAdapter, Detected_KinematicsFlags_AllSetWithAllFields)
{
  const auto [detected, tracked] = convert_one(make_adapter(), make_radar_object());

  EXPECT_TRUE(detected.kinematics.has_position_covariance);
  EXPECT_TRUE(detected.kinematics.has_twist);
  EXPECT_TRUE(detected.kinematics.has_twist_covariance);
  EXPECT_EQ(detected.kinematics.orientation_availability, DetectedObjectKinematics::AVAILABLE);
  EXPECT_EQ(tracked.kinematics.orientation_availability, TrackedObjectKinematics::AVAILABLE);
}

// The tracked object's UUID: the radar's 32-bit object id, least significant byte first, then
// the 8 bytes of the input topic name's hash, least significant byte first, then 4 zero bytes.
// The hash keeps the ids of two radars on different topics apart when their tracks are merged.
TEST(RadarObjectsAdapter, Tracked_Uuid_ObjectIdThenTopicHashThenZeros)
{
  RadarObject radar = make_radar_object();
  radar.object_id = 0x8899AABBu;

  const auto [detected, tracked] = convert_one(make_adapter(), radar);

  const std::array<std::uint8_t, 8> hash = topic_hash_bytes(topic_name);
  const std::array<std::uint8_t, 16> expected = {
    0xBB,    0xAA,    0x99,    0x88,    hash[0], hash[1], hash[2], hash[3],
    hash[4], hash[5], hash[6], hash[7], 0,       0,       0,       0};
  EXPECT_EQ(tracked.object_id.uuid, expected);
}

// Two adapters on different topics give the same object different ids.
TEST(RadarObjectsAdapter, Tracked_Uuid_DiffersBetweenTopics)
{
  RadarObjectsAdapter other(make_params(), identity_remap(), "/sensing/radar/rear/objects_raw");
  other.update_radar_info(make_radar_info(required_fields));
  const RadarObject radar = make_radar_object();

  const auto [detected, tracked] = convert_one(make_adapter(), radar);
  const auto [other_detected, other_tracked] = convert_one(other, radar);

  EXPECT_NE(tracked.object_id.uuid, other_tracked.object_id.uuid);
}

// Only MOVEMENT_STATUS_DYNAMIC counts as moving.
TEST(RadarObjectsAdapter, Tracked_Stationary_OnlyDynamicIsMoving)
{
  const RadarObjectsAdapter adapter = make_adapter();
  const std::vector<std::pair<std::uint8_t, bool>> cases = {
    {RadarObject::MOVEMENT_STATUS_DYNAMIC, false},
    {RadarObject::MOVEMENT_STATUS_STATIC, true},
    {RadarObject::MOVEMENT_STATUS_INVALID, true},
    {RadarObject::MOVEMENT_STATUS_UNKNOWN, true},
  };
  for (const auto & [status, stationary] : cases) {
    SCOPED_TRACE(static_cast<int>(status));
    RadarObject radar = make_radar_object();
    radar.movement_status = status;

    const auto [detected, tracked] = convert_one(adapter, radar);

    EXPECT_EQ(tracked.kinematics.is_stationary, stationary);
  }
}

// The tracked object shares with the detected one everything but its identity, its acceleration
// and the stationary flag.
TEST(RadarObjectsAdapter, Tracked_SharedFields_MatchDetected)
{
  const auto [detected, tracked] = convert_one(make_adapter(), facing(make_radar_object(), 0.3));

  EXPECT_EQ(tracked.existence_probability, detected.existence_probability);
  EXPECT_EQ(tracked.classification, detected.classification);
  EXPECT_EQ(tracked.kinematics.pose_with_covariance, detected.kinematics.pose_with_covariance);
  EXPECT_EQ(tracked.kinematics.twist_with_covariance, detected.kinematics.twist_with_covariance);
  EXPECT_EQ(tracked.shape, detected.shape);
}

// ---------------------------------------------------------------------------------------------
// Classification: one entry per radar classification, label looked up in the remap, probability
// copied, order kept; a label without an entry becomes UNKNOWN. Both outputs get the same list.
// ---------------------------------------------------------------------------------------------

TEST(RadarObjectsAdapter, Classification_RemapApplied_OrderAndProbabilitiesKept)
{
  RadarObjectsAdapter adapter(
    make_params(),
    ClassificationRemap{
      {RadarClassification::CAR, ObjectClassification::CAR},
      {RadarClassification::MOTORCYCLE, ObjectClassification::CAR},
      {RadarClassification::HAZARD, ObjectClassification::UNKNOWN}},
    topic_name);
  adapter.update_radar_info(make_radar_info(required_fields));
  RadarObject radar = make_radar_object();
  radar.classifications = {
    make_classification(RadarClassification::MOTORCYCLE, 0.3f),
    make_classification(RadarClassification::CAR, 0.8f),
    make_classification(RadarClassification::HAZARD, 0.05f),
    make_classification(RadarClassification::BUS, 0.1f),             // no entry
    make_classification(RadarClassification::OVER_DRIVABLE, 0.2f),   // no entry
    make_classification(RadarClassification::UNDER_DRIVABLE, 0.4f),  // no entry
  };

  const auto [detected, tracked] = convert_one(adapter, radar);

  const std::vector<LabeledProbability> expected = {
    {ObjectClassification::CAR, 0.3f},      {ObjectClassification::CAR, 0.8f},
    {ObjectClassification::UNKNOWN, 0.05f}, {ObjectClassification::UNKNOWN, 0.1f},
    {ObjectClassification::UNKNOWN, 0.2f},  {ObjectClassification::UNKNOWN, 0.4f}};
  EXPECT_EQ(labeled_probabilities(detected.classification), expected);
  EXPECT_EQ(tracked.classification, detected.classification);
}

TEST(RadarObjectsAdapter, Classification_NoClassifications_EmptyList)
{
  RadarObject radar = make_radar_object();
  radar.classifications.clear();

  const auto [detected, tracked] = convert_one(make_adapter(), radar);

  EXPECT_TRUE(detected.classification.empty());
  EXPECT_TRUE(tracked.classification.empty());
}

// ---------------------------------------------------------------------------------------------
// The remap table built from the parameter names (make_classification_remap_from_string_pair),
// and what it does to the classifications once the adapter is given it.
// ---------------------------------------------------------------------------------------------

namespace
{
// The ten radar labels the parameters can name, with their ids.
const std::map<std::string, RadarClassification::_label_type> radar_labels = {
  {"UNKNOWN", RadarClassification::UNKNOWN}, {"CAR", RadarClassification::CAR},
  {"TRUCK", RadarClassification::TRUCK},     {"BUS", RadarClassification::BUS},
  {"TRAILER", RadarClassification::TRAILER}, {"MOTORCYCLE", RadarClassification::MOTORCYCLE},
  {"BICYCLE", RadarClassification::BICYCLE}, {"PEDESTRIAN", RadarClassification::PEDESTRIAN},
  {"ANIMAL", RadarClassification::ANIMAL},   {"HAZARD", RadarClassification::HAZARD}};

// The nine perception labels a parameter value can name, with their ids.
const std::map<std::string, ObjectClassification::_label_type> perception_labels = {
  {"UNKNOWN", ObjectClassification::UNKNOWN}, {"CAR", ObjectClassification::CAR},
  {"TRUCK", ObjectClassification::TRUCK},     {"BUS", ObjectClassification::BUS},
  {"TRAILER", ObjectClassification::TRAILER}, {"MOTORCYCLE", ObjectClassification::MOTORCYCLE},
  {"BICYCLE", ObjectClassification::BICYCLE}, {"PEDESTRIAN", ObjectClassification::PEDESTRIAN},
  {"ANIMAL", ObjectClassification::ANIMAL}};

// The table and what was reported while building it, together.
struct MadeRemap
{
  ClassificationRemap remap;
  NameMap unknown_perception_labels;
};

MadeRemap make_remap(const NameMap & names)
{
  MadeRemap made;
  made.remap = make_classification_remap_from_string_pair(names, made.unknown_perception_labels);
  return made;
}

// The shipped parameter file, as the node reads it.
const NameMap shipped_configuration = {
  {"UNKNOWN", "UNKNOWN"}, {"CAR", "CAR"},        {"TRUCK", "TRUCK"}, {"BUS", "CAR"},
  {"TRAILER", "CAR"},     {"MOTORCYCLE", "CAR"}, {"BICYCLE", "CAR"}, {"PEDESTRIAN", "PEDESTRIAN"},
  {"ANIMAL", "ANIMAL"},   {"HAZARD", "UNKNOWN"}};

// An adapter with a valid radar info and the remap built from `names`.
RadarObjectsAdapter make_adapter_with_remap(const NameMap & names)
{
  RadarObjectsAdapter adapter(make_params(), make_remap(names).remap, topic_name);
  adapter.update_radar_info(make_radar_info(required_fields));
  return adapter;
}
}  // namespace

// No names in, no entries out.
TEST(RadarObjectsAdapter, RemapTable_Empty_NoEntries)
{
  const auto [remap, unknown_perception_labels] = make_remap({});

  EXPECT_TRUE(remap.empty());
  EXPECT_TRUE(unknown_perception_labels.empty());
}

// Every radar label name can be a key and every perception label name can be a value; the
// table holds the ids of both. Radar labels the parameters do not name get no entry.
TEST(RadarObjectsAdapter, RemapTable_KnownNames_MappedToIds)
{
  for (const auto & [radar_name, radar_id] : radar_labels) {
    for (const auto & [perception_name, perception_id] : perception_labels) {
      SCOPED_TRACE(radar_name + " -> " + perception_name);

      const auto [remap, unknown_perception_labels] = make_remap({{radar_name, perception_name}});

      EXPECT_EQ(remap, (ClassificationRemap{{radar_id, perception_id}}));
      EXPECT_TRUE(unknown_perception_labels.empty());
    }
  }
}

// The shipped configuration remaps BUS, TRAILER, MOTORCYCLE and BICYCLE to CAR, because the
// radar in use does not tell these from cars reliably, and HAZARD to UNKNOWN. Applied by the
// adapter, the remap works entry by entry: an object that carries a CAR, a BUS, a MOTORCYCLE and
// a BICYCLE probability comes out with four CAR entries, each with its own probability, while
// the labels the configuration maps to themselves stay as they are.
TEST(RadarObjectsAdapter, RemapTable_ShippedConfiguration_LargeVehiclesAndTwoWheelersBecomeCar)
{
  const RadarObjectsAdapter adapter = make_adapter_with_remap(shipped_configuration);
  RadarObject radar = make_radar_object();
  radar.classifications = {
    make_classification(RadarClassification::CAR, 0.5f),
    make_classification(RadarClassification::TRUCK, 0.1f),
    make_classification(RadarClassification::BUS, 0.4f),
    make_classification(RadarClassification::MOTORCYCLE, 0.8f),
    make_classification(RadarClassification::BICYCLE, 0.3f),
    make_classification(RadarClassification::PEDESTRIAN, 0.02f),
    make_classification(RadarClassification::HAZARD, 0.05f)};

  const auto [detected, tracked] = convert_one(adapter, radar);

  const std::vector<LabeledProbability> expected = {
    {ObjectClassification::CAR, 0.5f},     {ObjectClassification::TRUCK, 0.1f},
    {ObjectClassification::CAR, 0.4f},     {ObjectClassification::CAR, 0.8f},
    {ObjectClassification::CAR, 0.3f},     {ObjectClassification::PEDESTRIAN, 0.02f},
    {ObjectClassification::UNKNOWN, 0.05f}};
  EXPECT_EQ(labeled_probabilities(detected.classification), expected);
}

// A perception label name the table does not know is not an error: the radar label maps to
// UNKNOWN, and the pair is reported so that the node can warn about it.
TEST(RadarObjectsAdapter, RemapTable_UnknownPerceptionName_MapsToUnknownAndReported)
{
  const auto [remap, unknown_perception_labels] =
    make_remap({{"CAR", "SPACESHIP"}, {"TRUCK", "TRUCK"}, {"BICYCLE", "bicycle"}});

  const ClassificationRemap expected = {
    {RadarClassification::CAR, ObjectClassification::UNKNOWN},
    {RadarClassification::TRUCK, ObjectClassification::TRUCK},
    {RadarClassification::BICYCLE, ObjectClassification::UNKNOWN}};
  EXPECT_EQ(remap, expected);
  // Reported by radar label, so that each entry is reported once
  const NameMap reported = {{"BICYCLE", "bicycle"}, {"CAR", "SPACESHIP"}};
  EXPECT_EQ(unknown_perception_labels, reported);

  // and, applied, the object's CAR comes out as UNKNOWN with its probability.
  RadarObject radar = make_radar_object();
  radar.classifications = {make_classification(RadarClassification::CAR, 0.8f)};
  const auto [detected, tracked] =
    convert_one(make_adapter_with_remap({{"CAR", "SPACESHIP"}}), radar);
  const std::vector<LabeledProbability> applied = {{ObjectClassification::UNKNOWN, 0.8f}};
  EXPECT_EQ(labeled_probabilities(detected.classification), applied);
}

// The radar labels the parameters can name are the ten above. The two radar-only labels
// (OVER_DRIVABLE, UNDER_DRIVABLE) are not among them, and naming one as a key is an error rather
// than a silent entry: the parameter set is fixed.
TEST(RadarObjectsAdapter, RemapTable_UnknownRadarName_Throws)
{
  for (const auto & name : {"OVER_DRIVABLE", "UNDER_DRIVABLE", "car", ""}) {
    SCOPED_TRACE(name);
    EXPECT_THROW(make_remap({{name, "CAR"}}), std::out_of_range);
  }
}
