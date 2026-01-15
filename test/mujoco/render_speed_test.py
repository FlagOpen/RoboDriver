import mujoco
import mujoco.viewer
import time

XML_PATH = "./descriptions/agilex_aloha/scene.xml"

# 加载模型
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 设置相机
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)

# 创建渲染器
renderer = mujoco.Renderer(model, height=1200, width=1600)
viewer = mujoco.viewer.launch_passive(model, data)

# 渲染统计
render_times = []
last_render_time = time.time()
render_interval = 1/30.0  # 30 FPS

print("开始模拟，按ESC退出...")

try:
    while viewer.is_running():
        # 物理模拟
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # 控制渲染频率（与模拟解耦）
        current_time = time.time()
        if current_time - last_render_time >= render_interval:
            # 测量渲染耗时
            render_start = time.time()
            
            renderer.update_scene(data, cam)
            pixels = renderer.render()
            
            render_end = time.time()
            render_time = render_end - render_start
            render_times.append(render_time)
            
            # 打印渲染统计（每100帧）
            if len(render_times) >= 100:
                avg_time = sum(render_times) / len(render_times)
                print(f"平均渲染耗时: {avg_time*1000:.1f}ms (最近100帧)")
                render_times.clear()
            
            last_render_time = current_time
        
        # 精确时间控制
        elapsed = time.time() - step_start
        time_until_next_step = model.opt.timestep - elapsed
        
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\n模拟被中断")
finally:
    # 最终统计
    if render_times:
        avg_time = sum(render_times) / len(render_times)
        print(f"最终平均渲染耗时: {avg_time*1000:.1f}ms")
    
    renderer.close()
    viewer.close()
    print("资源已释放")