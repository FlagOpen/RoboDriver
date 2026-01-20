import mujoco
import mujoco.viewer
import cv2
import time
import os
import threading
import numpy as np

XML_PATH = os.getenv("XML_PATH", "./descriptions/agilex_aloha/scene.xml")

# 共享数据
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
lock = threading.Lock()  # 数据锁
running = True  # 控制标志

def simulation_loop():
    """仿真线程函数"""
    global data, running
    
    while running:
        with lock:
            # 执行物理仿真步进
            mujoco.mj_step(model, data)
        time.sleep(0.001)  # 控制仿真频率

def viewer_loop():
    """Viewer和Render线程函数"""
    global data, running
    
    # 1. 创建渲染器 (必须在这个线程中创建)
    renderer = mujoco.Renderer(model, height=1200, width=1600)
    
    # 2. 创建并运行viewer (必须在同一个线程)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer started in thread:", threading.current_thread().name)
        
        # 配置相机
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        
        while viewer.is_running() and running:
            with lock:
                # 更新viewer显示
                viewer.sync()
                
                # 在同一线程中进行额外的render操作
                renderer.update_scene(data, camera=cam)
                image = renderer.render()
                
                # 可以保存图像或进行其他处理
                cv2.imshow('Render', image)
                cv2.waitKey(1)
            
            time.sleep(0.016)  # ~60Hz刷新率
    
    # 清理
    running = False

def main():
    """主线程"""
    global running
    print("Main thread:", threading.current_thread().name)
    
    # 启动仿真线程
    sim_thread = threading.Thread(target=simulation_loop, name="SimThread")
    sim_thread.daemon = True
    
    # 启动viewer/render线程
    viewer_thread = threading.Thread(target=viewer_loop, name="ViewerThread")
    viewer_thread.daemon = True
    
    sim_thread.start()
    viewer_thread.start()
    
    try:
        # 主线程等待
        while running:
            # 可以在这里处理其他逻辑
            time.sleep(0.1)
            
            # 检查线程是否存活
            if not sim_thread.is_alive() or not viewer_thread.is_alive():
                break
                
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        # 停止所有线程
        running = False
        
        # 等待线程结束
        sim_thread.join(timeout=1.0)
        viewer_thread.join(timeout=1.0)
        
        print("All threads stopped")

if __name__ == "__main__":
    main()