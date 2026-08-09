# RoboDriver-Robot-Agilex-Aloha-Follower-Dora

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README_en.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README.md)

## Start

```
pip install -e .
```

## Hardware Configuration

- **Robot**: Agilex Aloha dual-arm system
- **Cameras**: 3x Orbbec cameras (top, right, left)
- **Arms**: 2x Piper arms (right and left)
- **Communication**: Dora nodes over CAN bus

## Dataflow

The robot uses Dora dataflow with the following nodes:
- `camera_top`, `camera_right`, `camera_left`: Orbbec cameras
- `piper_right`, `piper_left`: Piper arm controllers
- `agilex_aloha_follower_dora`: Main robot node

## TODO

- Robot Calibration
- Improve error handling
- Add more documentation

## Acknowledgment

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to Agilex Robotics 🤗, [Agilex Robotics](https://www.agilex.ai/).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).
- Thanks to Piper team 🤗, [Piper](https://github.com/your-piper-repo).

## Cite

<!-- Add citation information here -->
