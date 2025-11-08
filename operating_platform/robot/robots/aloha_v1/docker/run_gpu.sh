#!/bin/bash

# 允许容器访问 X server
xhost +local:root  

# 1. 停止并删除已有容器（如果存在）
echo "清理旧容器（如果存在）..."
docker stop operating_platform 2>/dev/null || true
docker rm operating_platform 2>/dev/null || true

# 2. 创建并启动新容器，挂载当前目录到容器的 /app 目录
echo "正在创建并启动容器，并挂载当前目录..."
docker run -it \
  --name operating_platform \
  --privileged \
  --gpus all \
  --network host \
  -v "$(pwd)":/Operating-Platform \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp \
  -e http_proxy=http://127.0.0.1:7897 \
  -e https_proxy=http://127.0.0.1:7897 \
  -e DOROBOT_HOME=/root/DoRobot \
  operating-platform:V1.4_ubuntu20

# 3. 检查容器是否运行
if [ "$(docker inspect -f '{{.State.Running}}' operating_platform 2>/dev/null)" == "true" ]; then
  echo "容器已成功运行。"
else
  echo "容器启动失败！查看日志："
  docker logs operating_platform
  exit 1
fi

# 4. 进入容器（假设容器内使用 bash，否则改为 sh）
echo "正在进入容器..."
docker exec -it operating_platform