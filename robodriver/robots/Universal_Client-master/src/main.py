# src/main.py
import sys
import os
import time
import argparse
from src.utils.config_loader import ConfigLoader
from src.inference.cloud_agent import CloudInferenceAgent
from src.inference.data_collector_adapter import DataCollectionAdapter
from src.core.factory import create_connector  # 引入底层连接器工厂

def main():
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="LinkerBot Edge Node")
    parser.add_argument(
        '--collect', 
        action='store_true', 
        help='启动数据采集旁路 (Data Collection Adapter)'
    )
    parser.add_argument(
        '--inference', 
        action='store_true', 
        help='启动云端/本地推理控制 (Cloud Inference Agent)'
    )
    args = parser.parse_args()

    # 安全检查：至少启动一个功能
    if not args.collect and not args.inference:
        print("⚠️ 请至少指定一种运行模式: --collect 或 --inference (也可两者同时使用)")
        parser.print_help()
        return

    # 2. 加载配置
    try:
        config = ConfigLoader("config/settings.yaml").load()
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return

    collect_adapter = None
    agent = None
    robot = None  # 用于在无 inference 模式下保持底层存活

    try:
        # 3. 如果需要采集数据，启动采集旁路
        if args.collect:
            print("📊 启动数据采集模块...")
            collect_adapter = DataCollectionAdapter(config)
            collect_adapter.start()

        # 4. 根据是否开启推理，决定程序的运行主轴
        if args.inference:
            print("🤖 启动推理控制引擎 (将自动拉起底层硬件)...")
            agent = CloudInferenceAgent(config)
            # run() 是死循环，阻塞主线程
            agent.run() 
        else:
            # 如果没有启动 agent，说明没有人拉起底层 ROS/ZMQ 连接
            # 这里必须手动拉起底层连接，否则 collect 收集不到任何数据
            print("⚙️ 仅采集模式，初始化并拉起底层硬件连接...")
            robot = create_connector("config/settings.yaml")
            robot.start()
            
            print("✅ 系统就绪，正在持续采集数据 (按 Ctrl+C 退出)...")
            # 保持主线程存活
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 收到退出信号，正在优雅关闭所有组件...")
    except Exception as e:
        print(f"❌ 运行期发生致命错误: {e}")
    finally:
        # 5. 安全释放所有资源
        if collect_adapter:
            collect_adapter.stop()
        if robot:
            robot.stop()

if __name__ == "__main__":
    main()