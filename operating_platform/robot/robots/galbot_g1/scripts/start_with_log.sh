#!/bin/bash

# 配置参数
CONTAINER_NAME=dr-container-galbot-g1
PROJECT_DIR="/WanX-EI-Studio"
CONDA_ENV1="wanx-studio-galbot"
TIMEOUT=30  # 等待超时时间（秒）
CLEANED_UP=false
IN_CONTAINER_MODE=false  # 容器模式标志
OVERWRITE_LOGS=false     # 是否覆盖日志标志

# 修复：使用英文路径，不要中文引号！
HOST_HOME_DIR=/opt/wanx_studio
HOST_LOG_DIR="${HOST_HOME_DIR}/log"
HOST_CONFIG_DIR="${HOST_HOME_DIR}/config"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -c, --container      在容器内直接运行（容器模式）"
    echo "  -h, --host           在宿主机运行（控制容器） - 默认模式"
    echo "  -o, --overwrite-logs 覆盖过去的日志文件（清空旧日志）"
    echo "  -?, --help           显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0                     # 宿主机模式（默认），追加日志"
    echo "  $0 --container         # 容器模式，追加日志"
    echo "  $0 -o                  # 覆盖日志模式"
    echo "  $0 -c -o               # 容器模式 + 覆盖日志"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--container)
            IN_CONTAINER_MODE=true
            shift
            ;;
        -h|--host)
            IN_CONTAINER_MODE=false
            shift
            ;;
        -o|--overwrite-logs)
            OVERWRITE_LOGS=true
            shift
            ;;
        -\?|--help)
            show_help
            ;;
        *)
            echo "错误: 未知参数 '$1'" >&2
            echo "使用 '$0 --help' 查看帮助" >&2
            exit 1
            ;;
    esac
done

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

    # 容器模式下跳过容器相关操作
    if [ "$IN_CONTAINER_MODE" = false ]; then
        # 终止容器内的进程（如果容器存在）
        if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
            log "尝试终止容器内的关键进程..."

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

            # 移除未定义的 DATAFLOW_PATH_ROBOT_SO101 相关清理
            execute_in_container_with_result "pkill -f 'coordinator.py'" "终止 coordinator.py 进程"
            execute_in_container_with_result "pkill -f 'dora-daemon'" "终止 dora-daemon 守护进程"

            sleep 2
        fi
        
        # 停止容器
        if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
            log "停止容器 $CONTAINER_NAME..."
            docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
        fi
        
        # 恢复X11访问权限
        xhost - >/dev/null 2>&1 || true
    fi
}

trap cleanup INT TERM EXIT

# 检查容器是否存在
check_container() {
    if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        error "容器 $CONTAINER_NAME 不存在，请先创建容器"
    fi
}

wait_for_container() {
    log "等待容器就绪..."
    # 注释掉视频设备检查（如不需要）
    log "容器已启动（跳过视频设备检查）"
}

execute_in_container() {
    local cmd="$1"
    local log_file="$2"
    
    log "执行命令: $cmd"
    docker exec -t "$CONTAINER_NAME" bash -c "$cmd" \
        > >(tee -a "$log_file") 2>&1 &
    echo $! >> .pids
}

execute_local() {
    local cmd="$1"
    local log_file="$2"
    
    log "在容器内执行命令: $cmd"
    bash -c "$cmd" > >(tee -a "$log_file") 2>&1 &
    echo $! >> .pids
}

# 主程序
if [ "$IN_CONTAINER_MODE" = false ]; then
    log "运行模式: 宿主机模式（控制容器）"
    xhost + >/dev/null 2>&1 || error "无法设置 X11 转发"

    log "停止旧容器..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true

    log "启动容器..."
    docker start "$CONTAINER_NAME" || error "容器启动失败"

    check_container
    wait_for_container
else
    log "运行模式: 容器模式（直接在容器内运行）"
fi

# 创建主目录结构
mkdir -p "$HOST_HOME_DIR" "$HOST_LOG_DIR" "$HOST_CONFIG_DIR"

# 创建日志子目录
COORDINATOR_LOG_DIR="$HOST_LOG_DIR/coordinator"
DATAFLOW_LOG_DIR="$HOST_LOG_DIR/dataflow"
mkdir -p "$COORDINATOR_LOG_DIR" "$DATAFLOW_LOG_DIR"

# 处理日志覆盖
if [ "$OVERWRITE_LOGS" = true ]; then
    log "覆盖日志模式：清空旧日志文件"
    > "$COORDINATOR_LOG_DIR/last_run.log" 2>/dev/null || true
    > "$DATAFLOW_LOG_DIR/last_run.log" 2>/dev/null || true
else
    log "追加日志模式：将新日志追加到现有文件"
fi

# 获取机器编号
if [ -f "$HOST_CONFIG_DIR/machine_code" ]; then
    MACHINE_ID=$(cat "$HOST_CONFIG_DIR/machine_code")
else
    error "机器编号文件不存在: $HOST_CONFIG_DIR/machine_code"
fi

# 使用时间戳定义日志文件
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
COORDINATOR_LOG="$COORDINATOR_LOG_DIR/${MACHINE_ID}__${CURRENT_TIME}.log"
# DATAFLOW_LOG="$DATAFLOW_LOG_DIR/${MACHINE_ID}__${CURRENT_TIME}.log"  # 暂未使用

# 准备conda激活命令
CONDA_ACTIVATE="source /opt/conda/etc/profile.d/conda.sh && conda activate"

# 清理旧PID
rm -f .pids

# 选择执行函数
if [ "$IN_CONTAINER_MODE" = false ]; then
    EXEC_FUNCTION="execute_in_container"
else
    EXEC_FUNCTION="execute_local"
fi

log "启动协调器..."
$EXEC_FUNCTION "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV1 && python operating_platform/core/coordinator.py --robot.type=galbot_g1" "$COORDINATOR_LOG"

log "所有进程已启动，PID记录在.pids文件中"
log "监控日志文件："
echo "- $COORDINATOR_LOG"

# 等待所有后台进程
wait