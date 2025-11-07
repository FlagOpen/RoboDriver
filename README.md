# WanX-EI-Studio

Docs coming soon.

## Support Robot

P1:
```
aloha_v1, operating_platform <= 0.1.5
SO101_v1, operating_platform <= 0.1.5
galaxea_v1, operating_platform <= 0.1.5
realman_v1, operating_platform <= 0.1.5
dexterous_hand_v1, operating_platform <= 0.1.5
```

P2:
```
pika_v1, operating_platform <= 0.1.3
galbot_g1, operating_platform <= 0.1.3
leju_kuavo4p, operating_platform <= 0.1.3
agibot_a2d, operating_platform <= 0.1.3
```

P3:
```
adora_v1, Not Support
ruantong_v1, Not Support
```

## Start
creat conda env

```sh
conda create --name wanx-studio python==3.11
```

activate conda env

```sh
conda activate wanx-studio
```

install this project

```sh
pip install -e .
```

**install pytorch, according to your platform**

```sh
# ROCM 6.1 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.1
# ROCM 6.2.4 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

## TODO

- Validate SO101 in new Code
- Function: Compare server code version
- Save device info
