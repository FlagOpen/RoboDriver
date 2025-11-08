import os  
import time
import sys  
import numpy as np
from scipy.spatial.transform import Rotation
import pyarrow as pa  
from dora import Node    


from teleop import MocapTeleVisionWrapper





class TelepDataCollector:  
    def __init__(self, dora_node, teleop_collect):  
        self.dora_node = dora_node  
        self.teleop = teleop_collect
        self.flag=False
          
    def collect_data(self):  
        try:  
            # 调用 telep.py 的 getdata 函数获取所有数据 
            
            head_pose,left_wrist_pose,right_wrist_pose,left_finger_joints,right_finger_joints,left_full_skeleton,right_full_skeleton = self.teleop.get_data()
            
            if head_pose and left_finger_joints and left_wrist_pose and right_full_skeleton:
                self.flag=True

            # print(left_wrist_pose)
                
            if self.flag: 
                head_pose = np.array(head_pose, dtype=np.float32)
                left_wrist_pose = np.array(left_wrist_pose, dtype=np.float32)
                right_wrist_pose = np.array(right_wrist_pose, dtype=np.float32)
                left_finger_joints = np.array(left_finger_joints, dtype=np.float32)
                right_finger_joints= np.array(right_finger_joints, dtype=np.float32)
                left_full_skeleton = np.array(left_full_skeleton, dtype=np.float32)
                right_full_skeleton = np.array(right_full_skeleton, dtype=np.float32)
                # 发送到 Dora 数据流  
                self.dora_node.send_output("head_pose", pa.array(head_pose)) 
                self.dora_node.send_output("left_wrist_pose", pa.array(left_wrist_pose))  
                self.dora_node.send_output("right_wrist_pose", pa.array(right_wrist_pose))  
                self.dora_node.send_output("left_finger_joints", pa.array(left_finger_joints))  
                self.dora_node.send_output("right_finger_joints", pa.array(right_finger_joints))
                self.dora_node.send_output("left_full_skeleton", pa.array(left_full_skeleton))
                self.dora_node.send_output("right_full_skeleton", pa.array(right_full_skeleton))
                
                self.flag=False

            
        except Exception as e:  
            print(f"Error collecting telep data: {e}")  
    
  
def main():  
    dora_node = Node()
    teleop_collect=MocapTeleVisionWrapper()  
    collector = TelepDataCollector(dora_node,teleop_collect)  
      
    for event in dora_node:  
        if event["type"] == "INPUT" and event["id"] == "tick":  
            collector.collect_data()  
        elif event["type"] == "STOP":  
            break  
  
if __name__ == "__main__":  
    main()
