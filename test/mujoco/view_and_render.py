import mujoco
import mujoco.viewer
import cv2
import time
import os
from pathlib import Path

XML_PATH = os.getenv("XML_PATH", "./descriptions/agilex_aloha/scene.xml")

# 加载模型
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 设置相机（可选，否则用默认视角）
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)  # 或指定 track/body/camera

paused = False

def key_callback(keycode):
    global paused
    if chr(keycode) == ' ':
        paused = not paused

# 创建 offscreen renderer（用于 OpenCV）
height, width = 480, 640
renderer = mujoco.Renderer(model, height=height, width=width)
viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
# 启动 passive viewer（GUI）

while viewer.is_running():
    if not paused:
        mujoco.mj_step(model, data)
        
        # 同步到 GUI viewer
        viewer.sync()
        
        # 同步到 offscreen renderer 并显示在 OpenCV
        renderer.update_scene(data, cam)
        pixels = renderer.render()
        # cv2.imshow("Offscreen Render", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)  # 必须调用，否则窗口不刷新
    
    # 控制帧率（注意：不要过长 sleep，否则 GUI 卡顿）
    time.sleep(0.002)  # ~500 Hz，与物理步长匹配

# 自动清理 renderer（因为用了 with，但这里没用 with，所以手动 close）
renderer.close()
viewer.close()
# cv2.destroyAllWindows()