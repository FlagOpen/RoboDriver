import mujoco
import mujoco.viewer
import cv2
import time
import os
import threading
from collections import deque
import numpy as np

XML_PATH = os.getenv("XML_PATH", "./descriptions/agilex_aloha/scene.xml")

# 加载模型
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 设置相机
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)

# 创建渲染器和查看器
renderer = mujoco.Renderer(model, height=1200, width=1600)
viewer = mujoco.viewer.launch_passive(model, data)

# 线程安全的渲染队列
class RenderThread(threading.Thread):
    def __init__(self, model, camera):
        super().__init__()
        self.model = model
        self.camera = camera
        self.renderer = mujoco.Renderer(model, height=1200, width=1600)
        self.running = True
        self.data_queue = deque(maxlen=1)  # 只保留最新的数据
        self.render_times = []
        self.lock = threading.Lock()
        self.avg_render_time = 0
        self.frame_count = 0
        
    def update_data(self, new_data):
        """更新要渲染的数据（线程安全）"""
        with self.lock:
            # 深拷贝关键数据
            self.data_queue.append({
                'time': new_data.time,
                'qpos': new_data.qpos.copy() if new_data.qpos is not None else None,
                'qvel': new_data.qvel.copy() if new_data.qvel is not None else None,
                'xpos': new_data.xpos.copy() if new_data.xpos is not None else None,
                'xquat': new_data.xquat.copy() if new_data.xquat is not None else None,
            })
    
    def run(self):
        """渲染线程主循环"""
        print("渲染线程启动")
        
        while self.running:
            if self.data_queue:
                render_start = time.time()
                
                # 获取最新数据
                with self.lock:
                    latest_data = self.data_queue[-1]
                
                # 创建临时数据对象用于渲染
                temp_data = mujoco.MjData(self.model)
                if latest_data['qpos'] is not None:
                    temp_data.qpos[:] = latest_data['qpos']
                if latest_data['qvel'] is not None:
                    temp_data.qvel[:] = latest_data['qvel']
                if latest_data['xpos'] is not None:
                    temp_data.xpos[:] = latest_data['xpos']
                if latest_data['xquat'] is not None:
                    temp_data.xquat[:] = latest_data['xquat']
                temp_data.time = latest_data['time']
                
                # 向前运动学
                mujoco.mj_forward(self.model, temp_data)
                
                # 渲染
                self.renderer.update_scene(temp_data, self.camera)
                pixels = self.renderer.render()
                
                render_end = time.time()
                render_time = render_end - render_start
                
                # 更新统计
                self.render_times.append(render_time)
                self.frame_count += 1
                
                # 显示到OpenCV窗口（可选）
                if self.frame_count % 5 == 0:  # 每5帧显示一次
                    try:
                        img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                        cv2.imshow('MuJoCo Render', img_bgr)
                        cv2.waitKey(1)
                    except:
                        pass
                
                # 计算平均渲染时间（最近100帧）
                if len(self.render_times) > 100:
                    self.render_times.pop(0)
                self.avg_render_time = sum(self.render_times) / len(self.render_times) if self.render_times else 0
            
            # 控制渲染帧率（例如30FPS）
            time.sleep(1/30.0)  # 约33ms
    
    def stop(self):
        """停止渲染线程"""
        self.running = False
        self.join()
        cv2.destroyAllWindows()
        self.renderer.close()
        print("渲染线程已停止")

# 创建并启动渲染线程
render_thread = RenderThread(model, cam)
render_thread.start()

# 物理模拟主循环
print("物理模拟开始，按ESC退出查看器...")

# 性能统计
physics_times = []
physics_frame_count = 0
last_print_time = time.time()
target_fps = 1 / model.opt.timestep  # 根据timestep计算理论FPS

try:
    while viewer.is_running():
        physics_start = time.time()
        
        # 物理模拟步进
        mujoco.mj_step(model, data)
        
        # 同步查看器（可以与物理模拟保持同步）
        viewer.sync()
        
        # 更新渲染线程的数据（异步）
        render_thread.update_data(data)
        
        physics_end = time.time()
        physics_time = physics_end - physics_start
        physics_times.append(physics_time)
        physics_frame_count += 1
        
        # 每秒打印一次统计信息
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            # 物理模拟统计
            avg_physics_time = sum(physics_times) / len(physics_times) if physics_times else 0
            physics_fps = len(physics_times) / (current_time - last_print_time)
            
            # 渲染统计
            avg_render_time = render_thread.avg_render_time
            render_fps = render_thread.frame_count / (current_time - last_print_time)
            
            print(f"\n=== 性能统计 (过去1秒) ===")
            print(f"物理模拟: {physics_fps:.1f} FPS (目标: {target_fps:.1f} FPS)")
            print(f"物理步耗时: {avg_physics_time*1000:.2f}ms")
            print(f"渲染线程: {render_fps:.1f} FPS")
            print(f"渲染耗时: {avg_render_time*1000:.2f}ms")
            print(f"渲染队列深度: {len(render_thread.data_queue)}")
            print("="*30)
            
            # 重置统计
            physics_times = []
            render_thread.frame_count = 0
            last_print_time = current_time
        
        # 精确时间控制
        elapsed = time.time() - physics_start
        time_until_next_step = model.opt.timestep - elapsed
        
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        else:
            # 如果落后于实时，打印警告
            if elapsed > model.opt.timestep * 1.1:
                print(f"警告: 物理步时间({elapsed*1000:.1f}ms)超过步长时间({model.opt.timestep*1000:.1f}ms)")

except KeyboardInterrupt:
    print("\n用户中断模拟")
except Exception as e:
    print(f"\n发生错误: {e}")
finally:
    # 停止渲染线程
    render_thread.stop()
    
    # 计算最终统计数据
    if physics_frame_count > 0:
        print(f"\n=== 最终统计 ===")
        print(f"物理模拟总步数: {physics_frame_count}")
        if physics_times:
            avg_physics = sum(physics_times) / len(physics_times)
            print(f"平均物理步耗时: {avg_physics*1000:.2f}ms")
    
    # 清理查看器
    viewer.close()
    print("模拟已结束，所有资源已释放")