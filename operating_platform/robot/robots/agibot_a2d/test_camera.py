import time
import cv2
from a2d_sdk.robot import CosineCamera as Camera

# 初始化相机
cameras = [
    "head",
    "hand_left",
    "hand_right",
    "head_depth",
    "hand_left_fisheye",
    "hand_right_fisheye",
    "back_left_fisheye",
    "back_right_fisheye",
    "head_center_fisheye",
    "head_left_fisheye",
    "head_right_fisheye"
]

camera = Camera(cameras)

# 获取帧率并判断相机是否可用（假设 get_fps 返回 float，不可用时可能为 0 或 None）
def is_camera_available(cam_name: str, threshold: float = 1.0) -> bool:
    try:
        fps = camera.get_fps(cam_name)
        print(f"Camera recieved fps: {fps}")
        # return fps is not None and fps >= threshold
        image, timestamp = camera.get_latest_image(cam_name)
        print(f"Image reciev: {image} and time: {timestamp}")
        if image is not None:
            return True
        else:
            return False
    except Exception as e:
        print(f"检查相机 {cam_name} 可用性时出错: {e}")
        return False

# 检测并列出所有可用相机
print("正在检测可用相机...")
available_cameras = []
for cam in cameras:
    if is_camera_available(cam):
        available_cameras.append(cam)
        print(f"✅ 相机 '{cam}' 可用 (FPS: {camera.get_fps(cam):.2f})")
    else:
        print(f"❌ 相机 '{cam}' 不可用")

if not available_cameras:
    print("⚠️ 没有检测到任何可用相机，程序退出。")
    camera.close()
    exit()

# 默认使用第一个可用相机（你可以修改为用户选择）
selected_camera = available_cameras[0]
print(f"\n将显示相机: '{selected_camera}'\n")

try:
    print("按 'q' 键退出显示...")
    while True:
        image, timestamp = camera.get_latest_image(selected_camera)

        if image is not None:
            # 处理图像颜色空间（假设 SDK 返回 RGB）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image  # 灰度图或单通道

            cv2.imshow(f"Camera: {selected_camera}", image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("未获取到图像，跳过...")
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n用户中断")

finally:
    cv2.destroyAllWindows()
    camera.close()