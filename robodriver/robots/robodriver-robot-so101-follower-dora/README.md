# RoboDriver-Robot-SO101-Follower-Dora

## Start

```
pip install -e .
```

<!-- ```
dora build dora/dataflow.yml
``` -->

```

```

## Inference

SO101 inference uses this follower plugin (no leader arm required; the `so101-aio-dora` plugin waits for leader joint data on connect and is meant for teleoperation).

1. Start the dora dataflow for the SO101 follower arm and cameras.
2. Launch RoboDriver (no teleoperator needed for inference):

   ```bash
   robodriver-run --robot.type=so101-follower-dora
   ```

3. Trigger `start_inference` from RoboDriver-Server with `policy_type=bc`. The BC path extracts the 6 joint values as the policy state and maps camera keys `image_top_rgb` / `image_wrist` to `cam_top` / `cam_wrist` (dual-arm robots use `cam_top_left/right` + `cam_wrist_left/right`).
4. `reset_pose` returns joints to zero and opens the gripper by default (`gripper_reset_value=100`, fully open in the 0-100 normalized range).

## TODO

- Robot Calibrate

## Acknowledgment

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to TheRobotStudio 🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).

## Cite

