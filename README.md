![RoboDriver](assets/images/robodriver.png)

[![GitHub Issues](https://img.shields.io/github/issues/FlagOpen/RoboDriver)](https://github.com/FlagOpen/RoboDriver/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/FlagOpen/RoboDriver)](https://github.com/FlagOpen/RoboDriver/discussions)

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![Simplified Chinese README](https://img.shields.io/badge/简体中文-d9d9d9)](./README_zh.md)

# RoboDriver

## Overview

RoboDriver is the core driver-layer component of DataCollect and serves as the standardized robot access module within the [CoRobot](https://github.com/FlagOpen/CoRobot) data stack.

As shown above, RoboDriver acts as the device-side driver adaptation layer. [RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server) is the data/control bridge layer and channel router, and [RoboXStudio](https://ei2data.baai.ac.cn/home) is the cloud- or platform-side console and data management center.

RoboDriver documentation: [RoboDriver-Doc](https://flagopen.github.io/RoboDriver-Doc)

## Latest News

- [2025-12-01] RoboDriver project open sourced

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Contributing](#contributing)
5. [Support](#support)
6. [License and Acknowledgements](#license-and-acknowledgements)
7. [Citation](#citation)

## Key Features

- **Multiple Robot Integration Methods**: RoboDriver supports integration beyond SDKs, including ROS and Dora.
- **LeRobot Compatibility**: RoboDriver's robot interface directly uses LeRobot's `Robot` class, which means RoboDriver and LeRobot are mutually compatible.
- **Enhanced LeRobot Dataset Format**: Different data structures are used at different stages of data handling. Data is stored as individual entries at the collection end for easier editing and transmission. The format also extends the original LeRobot specification.

## Quick Start

Please refer to the project documentation: [RoboDriver-Doc](https://flagopen.github.io/RoboDriver-Doc)

## Contributing

We sincerely welcome *any form of contribution* from the community. Whether it's **pull requests for new features**, **bug reports**, or even **small suggestions** to make RoboDriver more user-friendly—we deeply appreciate all contributions!

## Support

- Please use GitHub [Issues](https://github.com/FlagOpen/RoboDriver/issues) to report bugs and request new features.
- Please use GitHub [Discussions](https://github.com/FlagOpen/RoboDriver/discussions) to share ideas and ask questions.

## License and Acknowledgements

RoboDriver's source code is licensed under the Apache 2.0 License. This project would not be possible without the following amazing open-source projects:

- Thanks to the LeRobot team for open-sourcing 🤗 [LeRobot](https://github.com/huggingface/lerobot). RoboDriver is built as an improvement upon LeRobot.
- Thanks to TheRobotStudio team for open-sourcing the SO-100 and SO-101 robot arms 🤗 [SO-101](https://github.com/TheRobotStudio/SO-ARM100). The SO-101 arm is used as a deployment example in this project.
- Thanks to the dora-rs team for open-sourcing their robotics framework 🤗 [dora](https://github.com/dora-rs/dora). This framework enables a novel integration method for robots in this project.

## Citation

```bibtex
@misc{RoboDriver,
  author = {RoboDriver Authors},
  title = {RoboDriver: A robot control and data acquisition framework},
  month = {November},
  year = {2025},
  url = {https://github.com/FlagOpen/RoboDriver}
}
```
