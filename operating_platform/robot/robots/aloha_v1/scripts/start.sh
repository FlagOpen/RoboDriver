#!/bin/bash

# read -p "请输入机器人的名称，可输入的有realman、aloha、pika: " robot_type
# CONTAINER_NAME="baai-flask-server"
# PROJECT_DIR="/root/Operating-Platform"
# CONDA_ENV1="op-robot-$robot_type"
# CONDA_ENV2="op"
# CONDA_ENV3="dr-view-rerun"

# 配置参数
CONTAINER_NAME="wanx-robot-aloha-container"
PROJECT_DIR="/root/WanX-EI-Studio"
CONDA_ENV1="op-robot-aloha"
CONDA_ENV2="op"
CONDA_ENV3="dr-view-rerun"
DATAFLOW_PATH="operating_platform/robot/robots/aloha_v1/dora/robot_aloha_dataflow.yml"
TIMEOUT=30  # 等待超时时间（秒）
CLEANED_UP=false

HOST_HOME_DIR="/opt/wanx_studio"
HOST_LOG_DIR="${HOST_HOME_DIR}/log"
HOST_CONFIG_DIR="${HOST_HOME_DIR}/config"

# 颜色定义
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;32m$1\033[0m"
}

error() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;31m错误: $1\033[0m" >&2
    exit 1
}

# 清理函数
cleanup() {
    if $CLEANED_UP; then
        log "清理已执行过，跳过重复清理"
        return
    fi
    CLEANED_UP=true

    log "正在清理..."
    # 终止本地记录的进程
    if [ -f ".pids" ]; then
        kill $(cat .pids 2>/dev/null) 2>/dev/null || true
        rm -f .pids
    fi

    # 终止容器内的进程（如果容器存在）
    # 检查容器是否存在
    if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
        log "尝试终止容器内的关键进程..."

        # 定义一个辅助函数来执行命令并输出结果
        execute_in_container_with_result() {
            local cmd="$1"
            local desc="$2"
            log "执行容器内命令: $desc"
            local result=$(docker exec "$CONTAINER_NAME" sh -c "$cmd" 2>&1)
            if [ $? -eq 0 ]; then
                log "$desc 成功：$result"
            else
                log "$desc 失败或未找到匹配进程（可能已退出）"
            fi
        }

        # 终止容器内相关进程
        execute_in_container_with_result "pkill -f '$DATAFLOW_PATH'" "终止 dora 数据流进程"
        execute_in_container_with_result "pkill -f 'coordinator.py'" "终止 coordinator.py 进程"
        execute_in_container_with_result "pkill -f 'dora-daemon'" "终止 dora-daemon 守护进程"

        # 可选：等待几秒确保进程优雅退出
        sleep 2
    fi
    
    # 停止容器（如果仍在运行）
    if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
        log "停止容器 $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
    fi
    
    # 恢复X11访问权限
    xhost - >/dev/null 2>&1 || true
}

# 捕获中断信号
trap cleanup INT TERM EXIT

# 检查容器是否存在
check_container() {
    if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        error "容器 $CONTAINER_NAME 不存在，请先创建容器"
    fi
}

# 等待容器就绪（检查视频设备）
wait_for_container() {
    log "等待容器就绪..."
    local start_time=$(date +%s)
    
    while ! docker exec "$CONTAINER_NAME" sh -c "ls /dev/video* >/dev/null 2>&1"; do
        sleep 1
        local current_time=$(date +%s)
        if (( current_time - start_time > TIMEOUT )); then
            error "等待视频设备挂载超时"
        fi
    done
    log "视频设备已就绪"
}

# 执行容器内命令
execute_in_container() {
    local cmd="$1"
    local log_file="$2"
    
    log "执行命令: $cmd"
    docker exec -t -e DISPLAY=:0 "$CONTAINER_NAME" bash -c "$cmd" \
        > >(tee -a "$log_file") 2>&1 &
    echo $! >> .pids
}

# 主程序
xhost + >/dev/null 2>&1 || error "无法设置 X11 转发"

log "停止旧容器..."
docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true

log "启动容器..."
docker start "$CONTAINER_NAME" || error "容器启动失败"

check_container
wait_for_container

# 准备conda激活命令
CONDA_ACTIVATE="source /opt/conda/etc/profile.d/conda.sh && conda activate"

# 清理旧的PID文件
rm -f .pids

# 检查主目录是否存在，不存在则创建
if [ ! -d "$HOST_HOME_DIR" ]; then
    echo "创建主目录: $HOST_HOME_DIR"
    mkdir -p "$HOST_HOME_DIR"
fi

# 检查log路径是否存在，不存在则创建
if [ ! -d "$HOST_LOG_DIR" ]; then
    echo "创建日志目录: $HOST_LOG_DIR"
    mkdir -p "$HOST_LOG_DIR"
else
    echo "日志目录已存在: $HOST_LOG_DIR"
fi

echo "目录检查完成"

# 获取当前时间戳
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")

# 获取机器编号
if [ -f "$HOST_CONFIG_DIR/machine_code" ]; then
    MACHINE_ID=$(cat "$HOST_CONFIG_DIR/machine_code")
else
    echo "错误：机器编号文件不存在 $HOST_CONFIG_DIR/machine_code"
    exit 1
fi

# 使用时间戳定义日志文件名
COORDINATOR_LOG="$HOST_LOG_DIR/coordinator/${MACHINE_ID}__${CURRENT_TIME}.log"
DATAFLOW_LOG="$HOST_LOG_DIR/dataflow/${MACHINE_ID}__${CURRENT_TIME}.log"
RERUN_LOG="$HOST_LOG_DIR/rerun/${MACHINE_ID}__${CURRENT_TIME}.log"


# 并行执行任务
log "启动数据流..."
execute_in_container "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV1 && dora run $DATAFLOW_PATH" "$DATAFLOW_LOG"

sleep 5  # 简单的依赖等待

log "启动协调器..."
execute_in_container "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV2 && python operating_platform/scripts/run.py --robot.type=aloha_v1" "$COORDINATOR_LOG"

sleep 2  # 简单的依赖等待

log "启动Rerun 3D View..."
execute_in_container "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV3 && cd test/piper && dora run arms_only_web.yml" "$RERUN_LOG"


log "打开Rerun 3D View - 请在浏览器打开"http://localhost:9060?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy""
# execute_in_container "xdg-open "http://localhost:9060?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy""
# execute_in_container "xdg-open "http://localhost:9060?url=ws://localhost:9876""

log "所有进程已启动，PID记录在.pids文件中"
log "监控日志文件："
echo "- $DATAFLOW_LOG"
echo "- $COORDINATOR_LOG"
echo "- $RERUN_LOG"

# 持续运行直到收到中断信号
wait