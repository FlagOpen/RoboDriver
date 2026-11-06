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

SO101 inference can use this lightweight follower plugin with no leader arm. The `so101-aio-dora` plugin also supports inference; start it with `--robot.require_leader=false` to skip its default leader-data check.

1. Start the dora dataflow for the SO101 follower arm and cameras.
2. Launch RoboDriver (no teleoperator needed for inference):

   ```bash
   robodriver-run --robot.type=so101-follower-dora
   ```

3. To use the platform LeRobot WebSocket service, trigger `start_inference` from RoboDriver-Server with:
   - `policy_type=lerobot`
   - `data_channel.url=ws://10.1.2.74:8003`
   - the task `prompt` and control `fps` required by the platform

   The LeRobot path sends full checkpoint feature names and HWC uint8 RGB images, and uses the returned `action_fps` when executing the action chunk. The SO101 alias `observation.images.image_top_rgb` maps to `observation.images.image_top` by default; `observation.images.image_wrist` is unchanged. Other models can override this with the top-level `camera_mapping` dictionary in `start_inference`. If the server enables authentication, pass `Authorization: Bearer <token>` in `data_channel.headers`.
4. Alternatively, `policy_type=bc` selects the previous BC path, which maps camera keys to `cam_top` / `cam_wrist`.
5. `reset_pose` returns to the median initial pose measured from the SO101 block-placement dataset: `[2.154, -103.385, 89.802, 74.857, -2.242, 0.784]` for shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, and gripper. The gripper starts nearly closed to match the dataset; pass `gripper_reset_value=100` to open it instead.

## TODO

- Robot Calibrate

## Acknowledgment

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to TheRobotStudio 🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).

## Cite

