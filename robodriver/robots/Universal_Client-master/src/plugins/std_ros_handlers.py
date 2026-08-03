def decode_joint_state(msg, meta):
    return {
        "name": list(msg.name),
        "position": list(msg.position),
        "velocity": list(msg.velocity) if msg.velocity else []
    }

def encode_joint_state(msg_cls, command_data, meta):
    msg = msg_cls()
    msg.name = meta.get('joint_names', command_data.get('names', []))
    msg.position = [float(x) for x in command_data.get('position', [])]
    return msg

def decode_pose_stamped(msg, meta):
    return {
        "position": {"x": msg.pose.position.x, "y": msg.pose.position.y, "z": msg.pose.position.z},
        "orientation": {"x": msg.pose.orientation.x, "y": msg.pose.orientation.y, "z": msg.pose.orientation.z, "w": msg.pose.orientation.w}
    }

def encode_pose_stamped(msg_cls, command_data, meta):
    msg = msg_cls()
    raw_data = command_data.get('position', [])
    if len(raw_data) >= 7:
        msg.pose.position.x = float(raw_data[0])
        msg.pose.position.y = float(raw_data[1])
        msg.pose.position.z = float(raw_data[2])
        msg.pose.orientation.x = float(raw_data[3])
        msg.pose.orientation.y = float(raw_data[4])
        msg.pose.orientation.z = float(raw_data[5])
        msg.pose.orientation.w = float(raw_data[6])
    return msg

def register(connector):
    """将标准 Handler 挂载到 connector"""
    connector.register_handler(
        "sensor_msgs.msg.JointState", 
        decode_fn=decode_joint_state, 
        encode_fn=encode_joint_state
    )
    connector.register_handler(
        "geometry_msgs.msg.PoseStamped", 
        decode_fn=decode_pose_stamped, 
        encode_fn=encode_pose_stamped
    )