def encode_follow_joint(msg_cls, command_data, meta):
    """ 
    专门为 lbot_arm_interfaces.msg.FollowJoint 写的打包逻辑
    """
    msg = msg_cls()
    msg.joints = [float(x) for x in command_data.get('position', [])]
    msg.follow = command_data.get('follow', True) 
    return msg

def register(connector):
    """将自定义 Handler 挂载到 connector"""
    connector.register_handler(
        "lbot_arm_interfaces.msg.FollowJoint", 
        encode_fn=encode_follow_joint
    )