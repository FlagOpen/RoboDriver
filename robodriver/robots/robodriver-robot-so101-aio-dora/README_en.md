# RoboDriver-Robot-SO101-AIO-Dora

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README_en.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README.md)

## Start

```
pip install -e .
```

<!-- ```
dora build dora/dataflow.yml
``` -->

```

```

## LeRobot inference

The AIO plugin can also run LeRobot inference without leader joint data:

```bash
robodriver-run --robot.type=so101_aio_dora --robot.require_leader=false
```

Trigger `start_inference` with `policy_type=lerobot`, `data_channel.url=ws://10.1.2.74:8003`, and the task `prompt`. The current ACT model directly uses AIO's `observation.images.image_top` and `observation.images.image_wrist`, so no camera mapping override is needed.

`require_leader=false` only skips RoboDriver's leader-data check; the leader node remains in `dora/dataflow.yml`. Disconnect the leader during inference only when it is safe for the arm. To remove the leader from startup entirely, remove its dataflow node and connections as well.

## TODO

- Robot Calibrate

## Acknowledgment

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to TheRobotStudio 🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).

## Cite
