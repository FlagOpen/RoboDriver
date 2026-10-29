#!/bin/bash

# 检查 can-utils 是否已安装（仅提示，不强制退出）
if ! command -v candump &> /dev/null; then
    echo -e "\e[33m警告: 未检测到 can-utils，无法使用 candump/cansend 等工具。\e[0m"
    echo "建议安装: apt update && apt install can-utils"
    echo "----------------------------------------"
fi

# 检查是否在容器内运行
if grep -qs 'docker\|lxc' /proc/1/cgroup; then
    echo -e "\e[33m检测到容器环境，需确保:\e[0m"
    echo "1. 使用 --network host 启动容器 (关键!)"
    echo "2. 已挂载 /sys 和 /dev 目录 (如: -v /sys:/sys:ro)"
    echo "----------------------------------------"
fi

# 查找所有 CAN 接口
CAN_INTERFACES=$(ip -br link show type can 2>/dev/null | awk '{print $1}')

if [ -z "$CAN_INTERFACES" ]; then
    echo -e "\e[31m错误: 未找到任何 CAN 接口。\e[0m"
    echo "请检查:"
    echo "1. CAN 设备驱动是否加载 (lsmod | grep can)"
    echo "2. 是否使用 --network host 启动容器"
    echo "3. 宿主机上是否存在 CAN 接口 (ip link show type can)"
    exit 1
fi

echo -e "\e[32m找到以下 CAN 接口:\e[0m"
for iface in $CAN_INTERFACES; do
    echo "  - $iface"
done
echo "----------------------------------------"

# 解析每个 CAN 接口的 USB 位置
for iface in $CAN_INTERFACES; do
    SYS_PATH="/sys/class/net/$iface/device"
    
    # 检查设备路径是否存在
    if [ ! -L "$SYS_PATH" ]; then
        echo -e "\e[33m警告: 接口 $iface 无设备关联路径 ($SYS_PATH 不存在)\e[0m"
        echo "  可能原因: 非 USB 设备 或 驱动未正确加载"
        continue
    fi

    # 获取物理设备路径
    PHYS_PATH=$(readlink -f "$SYS_PATH" 2>/dev/null)
    if [ -z "$PHYS_PATH" ]; then
        echo -e "\e[33m警告: 无法解析 $iface 的设备路径\e[0m"
        continue
    fi

    # 从路径提取 USB 位置信息
    USB_INFO=""
    if echo "$PHYS_PATH" | grep -q '/usb'; then
        # 方法1: 尝试提取标准 USB 路径 (如 1-1.2)
        USB_INFO=$(echo "$PHYS_PATH" | grep -o '[0-9]\+-[0-9.]\+' | head -1)
        
        # 方法2: 备用提取 (处理复杂路径)
        if [ -z "$USB_INFO" ]; then
            USB_INFO=$(echo "$PHYS_PATH" | grep -o 'usb[0-9]\+/\(.*\)' | cut -d/ -f2)
        fi
    fi

    # 输出结果
    if [ -n "$USB_INFO" ]; then
        echo -e "\e[36m接口 $iface\e[0m -> \e[1mUSB 位置: $USB_INFO\e[0m"
        # 附加物理路径信息 (调试用)
        echo "  物理路径: ${PHYS_PATH#/sys}"
    else
        echo -e "\e[33m接口 $iface\e[0m -> \e[1m非 USB 设备\e[0m"
        echo "  设备路径: ${PHYS_PATH#/sys}"
    fi
done

# 附加诊断信息
echo "----------------------------------------"
echo -e "\e[34m诊断建议:\e[0m"
echo "1. 验证 USB 设备: lsusb | grep -i can"
echo "2. 检查驱动加载: lsmod | grep -E 'usb|can'"
echo "3. 容器启动示例:"
echo "   docker run -it --privileged --network host -v /sys:/sys:ro your_image"