# import cv2

# # realsense
# try:
#     import pyrealsense2 as rs
#     HAS_realsense = True
# except ImportError:
#     HAS_realsense = False

# # zed
# try:
#     import pyzed.sl as sl
#     HAS_zed = True
# except ImportError:
#     HAS_zed = False

# # orbbec SDK
# try:
#     import pyorbbecsdk as ob
#     HAS_orbbec = True
# except ImportError:
#     HAS_orbbec = False

# # ------------------ 摄像头分配 ------------------
# # { cam_name: comp_type }
# camera_allocation = {}  

# def release_camera(cam_name):
#     if cam_name in camera_allocation:
#         camera_allocation[cam_name] = None

# def allocate_camera(cam_name, comp_type):
#     if cam_name not in camera_allocation or camera_allocation[cam_name] is None:
#         camera_allocation[cam_name] = comp_type
#         return True
#     return False

# # ------------------ 探测器 ------------------
# class CameraDetector:
#     def __init__(self, max_opencv_index=10):
#         self.max_opencv_index = max_opencv_index/hdas/camera_wrist_right/aligned_depth_to_color/image_raw
#             if cap.isOpened():
#                 cam_name = f"opencv_{i}"
#                 cameras.append(cam_name)
#                 if cam_name not in camera_allocation:
#                     camera_allocation[cam_name] = None
#             cap.release()
#         return cameras

#     def detect_realsense(self):  
#         if not HAS_realsense:  
#             return []  
#         ctx = rs.context()  
#         cameras = []  
#         device_list = ctx.query_devices() 
#         for dev in device_list:  
#             name = dev.get_info(rs.camera_info.name)  
#             serial = dev.get_info(rs.camera_info.serial_number)  
#             cam_name = f"realsense_{serial}"  
#             cameras.append(cam_name)  
#             if cam_name not in camera_allocation:  
#                 camera_allocation[cam_name] = None  
#         return cameras

#     def detect_orbbec(self):  
#         if not HAS_orbbec:  
#             return []  
#         cameras = []  
#         ctx = ob.Context()  
#         device_list = ctx.query_devices()  
#         for device in device_list:  
#             info = device.get_device_info()  
#             cam_name = f"orbbec_{info.get_serial_number()}"  
#             cameras.append(cam_name)  
#             if cam_name not in camera_allocation:  
#                 camera_allocation[cam_name] = None  
#         return cameras

#     def detect_zed(self):
#         if not HAS_zed:
#             return []
#         cameras = []
#         for info in sl.Camera.get_device_list():
#             cam_name = f"zed_{info.serial_number}"
#             cameras.append(cam_name)
#             if cam_name not in camera_allocation:
#                 camera_allocation[cam_name] = None
#         return cameras

#     def detect_all(self):
#         result = {}
#         opencv = self.detect_opencv()
#         if opencv:
#             result["opencv"] = {"count": len(opencv), "list": opencv}

#         rs_cams = self.detect_realsense()
#         if rs_cams:/hdas/camera_wrist_right/aligned_depth_to_color/image_raw
#             result["realsense"] = {"count": len(rs_cams), "list": rs_cams}

#         orbbec = self.detect_orbbec()
#         if orbbec:
#             result["orbbec"] = {"count": len(orbbec), "list": orbbec}

#         zed = self.detect_zed()
#         if zed:
#             result["zed"] = {"count": len(zed), "list": zed}

#         return result

# # ------------------ 对外接口 ------------------
# def get_available_cameras():
#     detector = CameraDetector()
#     all_cams = detector.detect_all()
#     #print(all_cams)
#     available = []
#     for cam_type, info in all_cams.items():
#         for cam_name in info["list"]:
#             if camera_allocation.get(cam_name) is None: 
#                 available.append(cam_name)
#     return available


# detect_components.py
import cv2
import subprocess
import sys

# realsense
try:
    import pyrealsense2 as rs
    HAS_realsense = True
except ImportError:
    HAS_realsense = False

# zed
try:
    import pyzed.sl as sl
    HAS_zed = True
except ImportError:
    HAS_zed = False

# orbbec SDK
try:
    import pyorbbecsdk as ob
    HAS_orbbec = True
except ImportError:
    HAS_orbbec = False

# ------------------ 摄像头分配 ------------------
# { cam_name: comp_type }
camera_allocation = {}

def release_camera(cam_name):
    """释放分配，支持被动存在的 key"""
    if cam_name in camera_allocation:
        camera_allocation[cam_name] = None

def allocate_camera(cam_name, comp_type):
    """尝试分配摄像头/话题。若cam_name不存在，会先建立记录并分配。"""
    if cam_name not in camera_allocation or camera_allocation[cam_name] is None:
        camera_allocation[cam_name] = comp_type
        return True
    return False

# ------------------ 探测器 ------------------
class CameraDetector:
    def __init__(self, max_opencv_index=10):
        self.max_opencv_index = max_opencv_index

    def detect_opencv(self):
        cameras = []
        for i in range(self.max_opencv_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cam_name = f"opencv_{i}"
                cameras.append(cam_name)
                if cam_name not in camera_allocation:
                    camera_allocation[cam_name] = None
            cap.release()
        return cameras

    def detect_realsense(self):
        if not HAS_realsense:
            return []
        ctx = rs.context()
        cameras = []
        device_list = ctx.query_devices()
        for dev in device_list:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            cam_name = f"realsense_{serial}"
            cameras.append(cam_name)
            if cam_name not in camera_allocation:
                camera_allocation[cam_name] = None
        return cameras

    def detect_orbbec(self):
        if not HAS_orbbec:
            return []
        cameras = []
        ctx = ob.Context()
        device_list = ctx.query_devices()
        for device in device_list:
            info = device.get_device_info()
            cam_name = f"orbbec_{info.get_serial_number()}"
            cameras.append(cam_name)
            if cam_name not in camera_allocation:
                camera_allocation[cam_name] = None
        return cameras

    def detect_zed(self):
        if not HAS_zed:
            return []
        cameras = []
        for info in sl.Camera.get_device_list():
            cam_name = f"zed_{info.serial_number}"
            cameras.append(cam_name)
            if cam_name not in camera_allocation:
                camera_allocation[cam_name] = None
        return cameras

    # 原有 hardware 检测的聚合（dora 模式）
    def detect_hardware_cameras(self):
        result = {}
        opencv = self.detect_opencv()
        if opencv:
            result["opencv"] = {"count": len(opencv), "list": opencv}

        rs_cams = self.detect_realsense()
        if rs_cams:
            result["realsense"] = {"count": len(rs_cams), "list": rs_cams}

        orbbec = self.detect_orbbec()
        if orbbec:
            result["orbbec"] = {"count": len(orbbec), "list": orbbec}

        zed = self.detect_zed()
        if zed:
            result["zed"] = {"count": len(zed), "list": zed}

        return result

    def detect_all(self, mode="dora"):
        """
        根据 mode 返回检测结果：
          - mode == "dora" : 返回硬件相关摄像头（opencv, realsense, orbbec, zed）
          - mode == "ros"  : 返回 ROS 话题检测结果（见 detect_ros_cameras()）
        """
        mode = (mode or "dora").lower()
        if mode == "ros2":
            # ros 模式下我们使用独立函数检测 ROS 话题
            ros_res = detect_ros2_cameras()
            # 将 camera_allocation 中为 ROS 话题预留条目（如果不存在就注册）
            for t in ros_res["image_list"] + ros_res["depth_image_list"]:
                if t not in camera_allocation:
                    camera_allocation[t] = None
            return {"ros2": {
                "image_count": len(ros_res["image_list"]),
                "depth_image_count": len(ros_res["depth_image_list"]),
                "image_list": ros_res["image_list"],
                "depth_image_list": ros_res["depth_image_list"]
            }}
        elif mode == "ros1":
            # ros 模式下我们使用独立函数检测 ROS 话题
            ros_res = detect_ros1_cameras()
            # 将 camera_allocation 中为 ROS 话题预留条目（如果不存在就注册）
            for t in ros_res["image_list"] + ros_res["depth_image_list"]:
                if t not in camera_allocation:
                    camera_allocation[t] = None
            return {"ros1": {
                "image_count": len(ros_res["image_list"]),
                "depth_image_count": len(ros_res["depth_image_list"]),
                "image_list": ros_res["image_list"],
                "depth_image_list": ros_res["depth_image_list"]
            }}
        else:
            return self.detect_hardware_cameras()

# ------------------ ROS 摄像头话题探测 ------------------
def detect_ros2_cameras():
    """
    使用 `rostopic list` 检测 ROS1 的话题。
    返回字典：{"image_list": [...], "depth_image_list": [...]}
    匹配规则（可根据需要调整）：
        - 包含 'camera'、'image'、'image_raw' 且不包含 'depth' -> 普通图像话题（image_list）
        - 包含 'camera'、'image'、'image_raw' 且包含 'depth' -> 深度图话题（depth_image_list）
    """
    cameras = {"image_list": [], "depth_image_list": []}
    try:
        # 调用 rostopic list
        proc = subprocess.run(["ros2","topic", "list"], capture_output=True, text=True, check=True)
        topics = proc.stdout.strip().splitlines()
    except Exception as e:
        # 当没有 roscore 或 rostopic 命令不可用时，直接返回空
        # 如果你希望在没有 roscore 时尝试 ROS master HTTP API，也可以在此扩展。
        # 现在保持简单明了。
        # print("[detect_ros_cameras] rostopic 调用失败:", e)
        return cameras

    for t in topics:
        t = t.strip()
        if not t:
            continue
        lower = t.lower()
        # 要求同时包含关键字 'camera' 和 'image'（并且包含 'image_raw' 更精确）
        if "image_raw" in lower:
            if "depth" in lower:
                cameras["depth_image_list"].append(t)
            else:
                cameras["image_list"].append(t)


    return cameras

def detect_ros1_cameras():
    """
    使用 `rostopic list` 检测 ROS1 的话题。
    返回字典：{"image_list": [...], "depth_image_list": [...]}
    匹配规则（可根据需要调整）：
        - 包含 'camera'、'image'、'image_raw' 且不包含 'depth' -> 普通图像话题（image_list）
        - 包含 'camera'、'image'、'image_raw' 且包含 'depth' -> 深度图话题（depth_image_list）
    """
    cameras = {"image_list": [], "depth_image_list": []}
    try:
        # 调用 rostopic list
        proc = subprocess.run(["rostopic", "list"], capture_output=True, text=True, check=True)
        topics = proc.stdout.strip().splitlines()
    except Exception as e:
        # 当没有 roscore 或 rostopic 命令不可用时，直接返回空
        # 如果你希望在没有 roscore 时尝试 ROS master HTTP API，也可以在此扩展。
        # 现在保持简单明了。
        # print("[detect_ros_cameras] rostopic 调用失败:", e)
        return cameras

    for t in topics:
        t = t.strip()
        if not t:
            continue
        lower = t.lower()
        # 要求同时包含关键字 'camera' 和 'image'（并且包含 'image_raw' 更精确）
        if "image_raw" in lower:
            if "depth" in lower:
                cameras["depth_image_list"].append(t)
            else:
                cameras["image_list"].append(t)


    return cameras

# ------------------ 对外接口 ------------------
def get_available_cameras(mode="dora"):
    """
    返回可用 '摄像头' 列表（names）。mode 可选 'dora' 或 'ros'。
    - dora: 返回硬件摄像头名字（opencv_0, realsense_..., etc）
    - ros:  返回 ROS image 话题名列表（只包含未被分配的）
    """
    detector = CameraDetector()
    all_detected = detector.detect_all(mode=mode)

    available = []
    if mode and mode.lower() == "ros1":
        ros_info = all_detected.get("ros1", {})
        image_list = ros_info.get("image_list", [])
        depth_list = ros_info.get("depth_image_list", [])
        for cam_name in image_list + depth_list:
            if camera_allocation.get(cam_name) is None:
                available.append(cam_name)
    elif mode and mode.lower() == "ros2":
        ros_info = all_detected.get("ros2", {})
        image_list = ros_info.get("image_list", [])
        depth_list = ros_info.get("depth_image_list", [])
        for cam_name in image_list + depth_list:
            if camera_allocation.get(cam_name) is None:
                available.append(cam_name)
    else:
        # hardware 模式：合并所有检测到的硬件列表
        for cam_type, info in all_detected.items():
            for cam_name in info.get("list", []):
                if camera_allocation.get(cam_name) is None:
                    available.append(cam_name)

    return available
