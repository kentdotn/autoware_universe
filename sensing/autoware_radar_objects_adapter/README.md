# autoware_radar_objects_adapter

## Purpose

This package converts `autoware_sensing_msgs::msg::RadarObjects` into `autoware_perception_msgs::msg::DetectedObjects` and `autoware_perception_msgs::msg::TrackedObjects`, acting as a simple integration of radars into the perception pipeline.

## RadarObjectsAdapter

A node that converts radar objects from the sensing definition into a perception friendly format with no filtering involved.

### Parameter: classification_remap

This parameter allows remapping of classification labels from `autoware_sensing_msgs::msg::RadarClassification` to `autoware_perception_msgs::msg::ObjectClassification`. It should be provided as a flat list of strings, where each pair of strings represents an input label and the corresponding output label.

For example, the current default configuration remaps `MOTORCYCLE` and `BICYCLE` from radar classification to `CAR` in the perception classification, while keeping other labels unchanged.

The remap serves two purposes:

- Labels the perception pipeline does not use (`HAZARD`, `OVER_DRIVABLE`, `UNDER_DRIVABLE`) are mapped to `UNKNOWN`. This holds for any radar.
- Labels a particular radar is known to report unreliably are corrected. The shipped configuration is tuned for the radar in use; it is not a property of radars in general and should be reviewed when another radar is connected.

**Note**: If multiple radar labels are remapped to the same perception label, multiple probabilities may appear for that label.
This does not violate any logic in Autoware but may be worth monitoring.

### Inputs / Outputs

#### Input

| Name               | Type                                     | Description                                |
| ------------------ | ---------------------------------------- | ------------------------------------------ |
| ~/input/objects    | autoware_sensing_msgs::msg::RadarObjects | Input radar objects as defined in sensing. |
| ~/input/radar_info | autoware_sensing_msgs::msg::RadarInfo    | Input radar info.                          |

#### Output

| Name                | Type                                           | Description                                                                                            |
| ------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| ~/output/detections | autoware_perception_msgs::msg::DetectedObjects | Output radar objects as detections.                                                                    |
| ~/output/tracks     | autoware_perception_msgs::msg::TrackedObjects  | Output radar objects as tracks, keeping the radar's object ids (made unique per input topic as UUIDs). |

## Parameters

{{ json_to_markdown("sensing/autoware_radar_objects_adapter/schema/radar_objects_adapter.schema.json") }}
