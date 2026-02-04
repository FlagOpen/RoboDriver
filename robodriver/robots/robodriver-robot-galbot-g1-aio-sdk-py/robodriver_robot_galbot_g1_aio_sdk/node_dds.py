# robodriver_robot_galbot_aio_ros2/node_dds_adapter.py



import os
import sys
import time
import json
import base64
import asyncio
import threading
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import websockets

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

# ---- Protobuf imports (keep consistent with manipulator.py) ----
# If your pb2 modules live elsewhere, adjust PYTHONPATH or edit the sys.path block below.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_this_file = os.path.abspath(__file__)
_pkg_dir = os.path.dirname(_this_file)
# Try to add sibling/parent paths like manipulator.py does (best-effort; safe if already in PYTHONPATH).
for rel in ["..", "../..", "./", "../galbot_msgs", "../../galbot_msgs"]:
    p = os.path.abspath(os.path.join(_pkg_dir, rel))
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

# These imports must match your robot-bridge protobuf type strings.
try:
    from galbot_msgs.sensor_proto import image_pb2
    from galbot_msgs.sensor_proto import camera_pb2
    from galbot_msgs.singorix_proto import sensor_pb2 as singorix_sensor_pb2
except Exception as e:
    raise ImportError(
        "Failed to import galbot protobuf modules. "
        "Make sure `galbot_msgs` is available in PYTHONPATH."
    ) from e


class AsyncLoopManager:
    """Run an asyncio loop in a background thread (same pattern as manipulator.py)."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        # Wait for loop ready
        for _ in range(20):
            if self.loop and self.loop.is_running():
                return
            time.sleep(0.05)
        raise RuntimeError("Async loop failed to start")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


class RobotSocket:
    """
    Receive robot messages over WebSocket, decode protobuf, and cache the latest:
      - latest_images[topic] = RGB ndarray (H,W,3)
      - arm_joint_data / gripper_data / leg_data / head_data = dicts keyed by joint name
      - latest_states[topic] includes timestamps
    """

    def __init__(self, robot_ip: str, bridge_port: int = 10800):
        self.robot_ip = robot_ip
        self.bridge_port = bridge_port
        self.uri = f"ws://{self.robot_ip}:{self.bridge_port}"

        self.loop_manager = AsyncLoopManager()
        self.task: Optional[asyncio.Task] = None
        self.ws = None

        # shared caches
        self.image_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_states: Dict[str, Dict[str, Any]] = {}

        self.arm_joint_data: Dict[str, Dict[str, Any]] = {"right_arm": {}, "left_arm": {}}
        self.gripper_data: Dict[str, Dict[str, Any]] = {"right_gripper": {}, "left_gripper": {}}
        self.leg_data: Dict[str, Dict[str, Any]] = {"leg": {}}
        self.head_data: Dict[str, Dict[str, Any]] = {"head": {}}

        self.protobuf_type_map = {
            "galbot.sensor_proto.CompressedImage": image_pb2.CompressedImage,
            "galbot.sensor_proto.CameraInfo": camera_pb2.CameraInfo,
            "galbot.singorix_proto.SingoriXSensor": singorix_sensor_pb2.SingoriXSensor,
        }

        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self.loop_manager.start()
        assert self.loop_manager.loop is not None
        self.task = asyncio.run_coroutine_threadsafe(self._listen(), self.loop_manager.loop)

    def stop(self):
        self._running = False
        try:
            if self.task:
                self.task.cancel()
        except Exception:
            pass
        self.loop_manager.stop()

    async def _listen(self):
        while self._running:
            try:
                async with websockets.connect(self.uri, ping_interval=10, ping_timeout=10) as ws:
                    self.ws = ws
                    while self._running:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        op = msg.get("op")
                        if op == "message":
                            await self._process_protobuf_message(msg)
                        # ignore heartbeat / others
            except Exception:
                # reconnect with small backoff
                await asyncio.sleep(0.2)

    async def _process_protobuf_message(self, message: Dict[str, Any]):
        topic = message.get("topic")
        type_str = message.get("type")
        data_b64 = message.get("data")
        if not all([topic, type_str, data_b64]):
            return

        pb_class = self.protobuf_type_map.get(type_str)
        if pb_class is None:
            return

        try:
            data = base64.b64decode(data_b64)
            pb_message = pb_class()
            pb_message.ParseFromString(data)

            # images
            if "CompressedImage" in type_str:
                if hasattr(pb_message, "data") and len(pb_message.data) > 0:
                    arr = np.frombuffer(pb_message.data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with self.image_lock:
                            self.latest_images[topic] = img

            # sensors
            if "singorix/wbcs/sensor" in topic:
                self._parse_and_store_joint_data(pb_message)

            with self.state_lock:
                self.latest_states[topic] = {
                    "message": pb_message,
                    "timestamp": int(message.get("pub_ts", 0)),
                    "received": int(time.time_ns()),
                    "type": type_str,
                }
        except Exception:
            return

    def _parse_and_store_joint_data(self, sensor_msg):
        if not getattr(sensor_msg, "joint_sensor_map", None):
            return

        for group_name, joint_sensor in sensor_msg.joint_sensor_map.items():
            n = len(joint_sensor.position)
            joint_data = {}
            for i in range(n):
                name = joint_sensor.name[i] if i < len(joint_sensor.name) else f"joint{i}"
                joint_data[name] = {
                    "position": joint_sensor.position[i] if i < len(joint_sensor.position) else 0.0,
                    "velocity": joint_sensor.velocity[i] if i < len(joint_sensor.velocity) else 0.0,
                    "effort": joint_sensor.effort[i] if i < len(joint_sensor.effort) else 0.0,
                    "current": joint_sensor.current[i] if i < len(joint_sensor.current) else 0.0,
                }

            if group_name == "right_arm":
                self.arm_joint_data["right_arm"] = joint_data
            elif group_name == "left_arm":
                self.arm_joint_data["left_arm"] = joint_data
            elif group_name == "right_gripper":
                self.gripper_data["right_gripper"] = joint_data
            elif group_name == "left_gripper":
                self.gripper_data["left_gripper"] = joint_data
            elif group_name == "leg":
                self.leg_data["leg"] = joint_data
            elif group_name == "head":
                self.head_data["head"] = joint_data


class GalbotDDSRosPublisher(Node):
    """
    Publish ROS topics expected by node.py without modifying node.py:
    NOTE: Defaults match manipulator.py (robot_ip=127.0.0.1, bridge_port=10800).
      - /camera/camera/color/image_raw  (sensor_msgs/Image)
      - /right_arm, /left_arm, /right_gripper, /left_gripper, /leg, /head (sensor_msgs/JointState)
    Optional:
      - /joint_states (mirror follower to leader, for bring-up)
    """

    def __init__(self):
        super().__init__("galbot_dds_to_ros_bridge")

        # ---- parameters ----
        self.declare_parameter("robot_ip", "127.0.0.1")
        self.declare_parameter("bridge_port", 10800)
        self.declare_parameter("publish_rate_hz", 30.0)

        # Which image source topic to use from DDS (substring match). If empty, use the first image seen.
        self.declare_parameter("dds_image_topic_hint", "")

        # Joint groups available: right_arm, left_arm, right_gripper, left_gripper, leg, head
        self.declare_parameter("joint_groups", ["right_arm", "left_arm", "right_gripper", "left_gripper", "leg", "head"])
        # NOTE: (disabled: /f_joint_states) is NOT published. `joint_groups` is only used for optional leader mirror (/joint_states).
        self.declare_parameter("publish_group_topics", True)
        # Publish per-group JointState topics. If empty, topics are "/<group>". If set (e.g. "(disabled: /f_joint_states)"), topics are "<prefix>/<group>".
        self.declare_parameter("group_topic_prefix", "")

        # Mirror (disabled: /f_joint_states) to /joint_states for leader (optional)
        self.declare_parameter("publish_leader_mirror", False)

        # ROS topics
        self.declare_parameter("ros_image_topic", "/camera/camera/color/image_raw")
        # Optional combined JointState topic (leader mirror)
        self.declare_parameter("ros_leader_joint_topic", "/joint_states")

        robot_ip = self.get_parameter("robot_ip").get_parameter_value().string_value
        bridge_port = int(self.get_parameter("bridge_port").value)
        rate = float(self.get_parameter("publish_rate_hz").value)

        self.dds_image_topic_hint = self.get_parameter("dds_image_topic_hint").value
        self.joint_groups: List[str] = list(self.get_parameter("joint_groups").value)
        self.publish_leader_mirror = bool(self.get_parameter("publish_leader_mirror").value)

        self.ros_image_topic = self.get_parameter("ros_image_topic").value
        self.ros_leader_joint_topic = self.get_parameter("ros_leader_joint_topic").value

        # ---- publishers ----
        self.pub_img = self.create_publisher(Image, self.ros_image_topic, 10)
        # Per-group joint state publishers (optional)
        self.group_pubs: Dict[str, Any] = {}
        if self.publish_group_topics:
            for g in ["right_arm","left_arm","right_gripper","left_gripper","leg","head"]:
                topic = f"/{g}" if not self.group_topic_prefix else f"{self.group_topic_prefix.rstrip('/')}/{g}"
                self.group_pubs[g] = self.create_publisher(JointState, topic, 10)
        self.pub_ljs = self.create_publisher(JointState, self.ros_leader_joint_topic, 10) if self.publish_leader_mirror else None

        # ---- socket ----
        self.sock = RobotSocket(robot_ip=robot_ip, bridge_port=bridge_port)
        self.sock.start()

        # publish timers
        period = 1.0 / max(rate, 1e-3)
        self.timer = self.create_timer(period, self._on_timer)

        # last-published guards
        self._last_img_received_ns: int = 0
        self._last_js_received_ns: int = 0

        self.get_logger().info(
            f"DDS->ROS bridge started. ws={self.sock.uri} "
            f"img->{self.ros_image_topic}, per-group joints->(/{right_arm,left_arm,right_gripper,left_gripper,leg,head})"
        )

    def destroy_node(self):
        try:
            self.sock.stop()
        except Exception:
            pass
        return super().destroy_node()

    def _pick_image(self) -> Optional[np.ndarray]:
        with self.sock.image_lock:
            if not self.sock.latest_images:
                return None

            if self.dds_image_topic_hint:
                for k, v in self.sock.latest_images.items():
                    if self.dds_image_topic_hint in k:
                        return v

            # fallback: pick the most recently received image topic (by latest_states received)
            with self.sock.state_lock:
                best_topic = None
                best_rcv = -1
                for t in self.sock.latest_images.keys():
                    st = self.sock.latest_states.get(t, {})
                    rcv = int(st.get("received", 0))
                    if rcv > best_rcv:
                        best_rcv = rcv
                        best_topic = t
                if best_topic is not None:
                    return self.sock.latest_images[best_topic]

            # fallback: any
            return next(iter(self.sock.latest_images.values()))

    def _gather_joint_dict_group(self, g: str) -> Dict[str, Dict[str, float]]:
        merged: Dict[str, Dict[str, float]] = {}
        def merge(src: Dict[str, Dict[str, float]]):
            for k, v in src.items():
                merged[k] = v
        if g == "right_arm":
            merge(self.sock.arm_joint_data.get("right_arm", {}))
        elif g == "left_arm":
            merge(self.sock.arm_joint_data.get("left_arm", {}))
        elif g == "right_gripper":
            merge(self.sock.gripper_data.get("right_gripper", {}))
        elif g == "left_gripper":
            merge(self.sock.gripper_data.get("left_gripper", {}))
        elif g == "head":
            merge(self.sock.head_data.get("head", {}))
        elif g == "leg":
            merge(self.sock.leg_data.get("leg", {}))
        return merged

    def _latest_sensor_received_ns(self) -> int:
        # best-effort: find newest "singorix/wbcs/sensor" received timestamp
        newest = 0
        with self.sock.state_lock:
            for topic, st in self.sock.latest_states.items():
                if "singorix/wbcs/sensor" in topic:
                    newest = max(newest, int(st.get("received", 0)))
        return newest

    def _latest_image_received_ns(self) -> int:
        newest = 0
        with self.sock.state_lock:
            for topic, st in self.sock.latest_states.items():
                if st.get("type") == "galbot.sensor_proto.CompressedImage":
                    newest = max(newest, int(st.get("received", 0)))
        return newest

    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # ---- publish JointState if new ----
        sensor_rcv = self._latest_sensor_received_ns()
        if sensor_rcv > self._last_js_received_ns:
            # Publish per-group topics (right_arm, left_arm, right_gripper, left_gripper, leg, head)
            if self.publish_group_topics and self.group_pubs:
                for g, pub in self.group_pubs.items():
                    gmap = self._gather_joint_dict_group(g)
                    if not gmap:
                        continue
                    gnames = list(gmap.keys())
                    gjs = JointState()
                    gjs.header.stamp = now
                    gjs.name = gnames
                    gjs.position = [float(gmap[n].get("position", 0.0)) for n in gnames]
                    gjs.velocity = [float(gmap[n].get("velocity", 0.0)) for n in gnames]
                    gjs.effort = [float(gmap[n].get("effort", 0.0)) for n in gnames]
                    pub.publish(gjs)

            # Optional: publish a combined JointState on /joint_states for compatibility/bring-up
            if self.pub_ljs is not None:
                merged = {}
                for g in self.joint_groups:
                    gmap = self._gather_joint_dict_group(g)
                    for k, v in gmap.items():
                        merged[k] = v
                if merged:
                    names = list(merged.keys())
                    js = JointState()
                    js.header.stamp = now
                    js.name = names
                    js.position = [float(merged[n].get("position", 0.0)) for n in names]
                    js.velocity = [float(merged[n].get("velocity", 0.0)) for n in names]
                    js.effort = [float(merged[n].get("effort", 0.0)) for n in names]
                    self.pub_ljs.publish(js)

            self._last_js_received_ns = sensor_rcv


        # ---- publish Image if new ----
        img_rcv = self._latest_image_received_ns()
        if img_rcv > self._last_img_received_ns:
            img = self._pick_image()
            if img is not None and img.ndim == 3 and img.shape[2] == 3:
                msg = Image()
                msg.header.stamp = now
                msg.height = int(img.shape[0])
                msg.width = int(img.shape[1])
                msg.encoding = "rgb8"
                msg.is_bigendian = 0
                msg.step = int(msg.width * 3)
                msg.data = img.tobytes()
                self.pub_img.publish(msg)
                self._last_img_received_ns = img_rcv


def main(args=None):
    rclpy.init(args=args)
    node = GalbotDDSRosPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
