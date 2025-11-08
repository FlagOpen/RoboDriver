#!/bin/bash

IMAGE_NAME=wanx-robot-aloha-image
IMAGE_VERSION=0.2
CONTAINER_NAME=wanx-robot-aloha-container
DOROBOT_HOME_HOST="/home/${USER}/DoRobot"
DOROBOT_HOME=/root/DoRobot


# 允许容器访问 X server
xhost +local:root  

# 1. 停止并删除已有容器（如果存在）
echo "清理旧容器（如果存在）..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true


# 2. 创建并启动新容器，挂载当前目录到容器的 /app 目录
echo "正在创建并启动容器，并挂载当前目录..."
docker run -it \
  --name $CONTAINER_NAME \
  --privileged \
  --network host\
  --shm-size=8g \
  -v $DOROBOT_HOME_HOST:$DOROBOT_HOME \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp \
  -e http_proxy=http://127.0.0.1:7897 \
  -e https_proxy=http://127.0.0.1:7897 \
  -e DOROBOT_HOME=$DOROBOT_HOME \
  $IMAGE_NAME:$IMAGE_VERSION

# 3. 检查容器是否运行
if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null)" == "true" ]; then
  echo "容器已成功运行。"
else
  echo "容器启动失败！查看日志："
  docker logs $CONTAINER_NAME
  exit 1
fi

# 4. 进入容器（假设容器内使用 bash，否则改为 sh）
echo "正在进入容器..."
docker exec -it $CONTAINER_NAME
