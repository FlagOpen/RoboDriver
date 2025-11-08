


import time
import zmq
import threading
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def matrix_to_pose(matrix):
        """将4x4矩阵转换为位置和四元数 (x,y,z, px,py,pz,pw)"""
        if matrix.shape != (4, 4):
            raise ValueError("输入必须是4x4齐次变换矩阵")
        
        x, y, z = matrix[:3, 3]
        
        rotation = R.from_matrix(matrix[:3, :3])
        px, py, pz, pw = rotation.as_quat()  # 返回顺序是(x,y,z,w)
        
        return (x, y, z, px, py, pz, pw)

class Manus_ZMQSubscriber:
    """
    Creates a thread that subscribes to a ZMQ publisher
    """
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)     
        self.socket.connect("tcp://localhost:8000")
        self._subscriber_thread = threading.Thread(target=self._update_value)
        self.left_glove_sn = "52f9d729"
        self.right_glove_sn = "4ef21d2c"
        self.updated = False
        self._subscriber_thread.start()
        self._joints = None
        self.last_message = None
        self._left_short_skeleton = None
        self._right_short_skeleton = None
        self._left_full_skeleton = None
        self._right_full_skeleton = None
    
    @property
    def left_joints(self):
        while self._joints is None:
            continue
        return self._joints[0:20] 
    
    @property
    def right_joints(self):
        while self._joints is None:
            continue
        return self._joints[20:40] 
    
    @property
    def left_short_skeleton(self):
        return self._left_short_skeleton
    
    @property
    def right_short_skeleton(self):
        return self._right_short_skeleton
    
    @property
    def left_full_skeleton(self):
        return self._left_full_skeleton
    
    @property
    def right_full_skeleton(self):
        return self._right_full_skeleton
    
    #This thread runs in the background and receives the messages
    def _update_value(self):
        while True:
            message = self.socket.recv()
            message = message.decode('utf-8')
            # print("message:", message)
            # print("time for manus: ", time.time())
            message = message.split(",") 
            if len(message) == 40:
                self._joints = list(map(float,message[0:40]))  #Get the right hand data (second half of the 40 datapoints coming in)
                self.updated = True
            elif len(message) == 176:
                if message[0] == self.left_glove_sn:
                    self._left_short_skeleton = self.parse_short_skeleton_and_send(message[0:176])
                    self._left_full_skeleton = self.parse_full_skeleton_and_send(message[0:176])
                elif message[0] == self.right_glove_sn:
                    self._right_short_skeleton = self.parse_short_skeleton_and_send(message[0:176])
                    self._right_full_skeleton = self.parse_full_skeleton_and_send(message[0:176])
                self.updated = True
            elif len(message) == 352:
                if message[0] == self.left_glove_sn:
                    self._left_short_skeleton = self.parse_short_skeleton_and_send(message[0:176])
                    self._left_full_skeleton = self.parse_full_skeleton_and_send(message[0:176])
                elif message[0] == self.right_glove_sn:
                    self._right_short_skeleton = self.parse_short_skeleton_and_send(message[0:176])
                    self._right_full_skeleton = self.parse_full_skeleton_and_send(message[0:176])
                if message[176] == self.left_glove_sn:
                    self._left_short_skeleton = self.parse_short_skeleton_and_send(message[176:352])
                    self._left_full_skeleton = self.parse_full_skeleton_and_send(message[176:352])
                elif message[176] == self.right_glove_sn:
                    self._right_short_skeleton = self.parse_short_skeleton_and_send(message[176:352])
                    self._right_full_skeleton = self.parse_full_skeleton_and_send(message[176:352])
                self.updated = True
    
    #If you set a flag in the C++ code, you can send all the data that comes from the raw skeleton of the glove.  This data is from thumb to pinky, across all joints from palm to fingertip.   This can slow things down though
    def parse_full_skeleton_and_send(self, data):
        full_skeleton_list = []
        # print(data[0])
        for i in range(0,25):
            position = [float(data[1 + i*7]), float(data[2 + i*7]), float(data[3 + i*7])]  #the first ID is right or left glove don't forget
            full_skeleton_list.append(position)
        if data[0] == self.left_glove_sn:
            return full_skeleton_list
        elif data[0] == self.right_glove_sn:
            return full_skeleton_list
        else:
            print("Glove serial number incorrect!")
            print(data[0])
            
    #This the dexcap style data, you only get the fingertip and the previous joint xyz as the data and then you can send that.  It goes from thumb_middle, thumb_tip, index_middle, index_tip etc.etc.
    def parse_short_skeleton_and_send(self, data):
        short_skeleton_list = []
        #short_idx = [3, 4, 8, 9, 13, 14, 18, 19, 23, 24] 
        ##Right now the integrated mode is in a different ordering, pinky, thumb, index, ring, middle
        ##Will be fixed to match the SDK in a future release
        short_idx = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15] 
        for i in short_idx:
            position = [float(data[1 + i*7]), float(data[2 + i*7]), float(data[3 + i*7])]  #the first ID is right or left glove don't forget
            short_skeleton_list.append(position)
        if data[0] == self.left_glove_sn:
            return short_skeleton_list
        elif data[0] == self.right_glove_sn:
            return short_skeleton_list
        else:
            print("Glove serial number incorrect!")  
            print(data[0])      


class Vive_ZMQSubscriber:
    """
    Creates a thread that subscribes to manus ZMQ publisher
    """
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        # 连接到发布者（请替换为实际 IP 地址，如 "tcp://192.168.1.100:5555"）
        self.socket.connect("tcp://192.168.123.162:5555")
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        # 订阅多个话题（分别订阅 right_elbow、left_elbow、chest）
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "right_elbow")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "left_elbow")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "chest")
        # self.socket.setsockopt(zmq.CONFLATE, True)     
        # self.socket.connect("tcp://localhost:8000")
        self.update=True

        self._subscriber_thread = threading.Thread(target=self._update_value)
        self._subscriber_thread.start()
        self._value = None
        self.last_message = None

        self.left_pos=None
        self.left_ori=None

        self.right_pos=None
        self.right_ori=None

        self.head_pos=None
        self.head_ori=None

        
    

    #This thread runs in the background and receives the messages
    def _update_value(self):
        while True:
            t1=time.time()
            # message = self.socket.recv()
            # print("message:", message)
            self.update=False
            # print("self.update before: ",self.update)
            
            message = self.socket.recv_string()
            # 拆分 topic 和 JSON 数据
            topic, json_data = message.split(' ', 1)
            pose = json.loads(json_data)
            
            # self.update=True
            # import pdb;pdb.set_trace()
          

            if topic == 'left_elbow':
                self.left_pos=pose["position"]
                self.left_ori=pose["orientation"]
                # print("left message: ",message)
                self.update=True
                # print("self.update after: ",self.update)
           
            elif topic == 'right_elbow':
                self.right_pos=pose["position"]
                self.right_ori=pose["orientation"]
                # print("right message: ",message)
            elif topic == 'chest':
                self.head_pos=pose["position"]
                self.head_ori=pose["orientation"]
                # print("head message: ", message)
            
            ##for debug\
            # dt=time.time()-t1
            # print("delta t is: ", dt)
            # self.update=True
            # print("self.update after: ",self.update)


            # message = message.decode('utf-8')
            # message = message.split(",") 
            # if len(message) == 40:
            #     self._value = list(map(float,message[0:20]))  #Get the left hand data (second half of the 40 datapoints coming in)



class MocapData:
    def __init__(self):
        self.manus_zmq_sub=Manus_ZMQSubscriber()
        self.vive_zmq_sub=Vive_ZMQSubscriber()

        # matrix=np.eye(4)
        # matrix[:3,3]=[self.vive_zmq_sub.head_pos['x'], self.vive_zmq_sub.head_pos['y'], self.vive_zmq_sub.head_pos['z']]
        # # matrix[:3,:3]=quaternion_matrix(self.vive_zmq_sub.head_ori)
        # quat = [self.vive_zmq_sub.head_ori['x'], self.vive_zmq_sub.head_ori['y'], self.vive_zmq_sub.head_ori['z'], self.vive_zmq_sub.head_ori['w']]
        # matrix[:3,:3]=R.from_quat(quat).as_matrix()

        # self.start_head_matrix=
    
    @property
    def head_matrix(self):
        matrix=np.eye(4)
        matrix[:3,3]=[self.vive_zmq_sub.head_pos['x'], self.vive_zmq_sub.head_pos['y'], self.vive_zmq_sub.head_pos['z']]
        # matrix[:3,:3]=quaternion_matrix(self.vive_zmq_sub.head_ori)
        quat = [self.vive_zmq_sub.head_ori['x'], self.vive_zmq_sub.head_ori['y'], self.vive_zmq_sub.head_ori['z'], self.vive_zmq_sub.head_ori['w']]
        matrix[:3,:3]=R.from_quat(quat).as_matrix()
        return matrix
    
    @property
    def left_hand(self):
        matrix=np.eye(4)
        # print("self.vive_zmq_sub.head_pos:", self.vive_zmq_sub.head_pos)
        matrix[:3,3]=[self.vive_zmq_sub.left_pos['x'], self.vive_zmq_sub.left_pos['y'], self.vive_zmq_sub.left_pos['z']]
        # matrix[:3,:3]=quaternion_matrix(self.vive_zmq_sub.left_ori)
        # print(np.array(self.vive_zmq_sub.left_ori))
        quat = [self.vive_zmq_sub.left_ori['x'], self.vive_zmq_sub.left_ori['y'], self.vive_zmq_sub.left_ori['z'], self.vive_zmq_sub.left_ori['w']]
        matrix[:3,:3]=R.from_quat(quat).as_matrix()

        # print("receive: ",self.vive_zmq_sub.update)
        return matrix


    @property
    def right_hand(self):
        matrix=np.eye(4)
        matrix[:3,3]= [self.vive_zmq_sub.right_pos['x'], self.vive_zmq_sub.right_pos['y'], self.vive_zmq_sub.right_pos['z']]
        # matrix[:3,:3]=quaternion_matrix(self.vive_zmq_sub.right_ori)
        quat = [self.vive_zmq_sub.right_ori['x'], self.vive_zmq_sub.right_ori['y'], self.vive_zmq_sub.right_ori['z'], self.vive_zmq_sub.right_ori['w']]
        matrix[:3,:3]=R.from_quat(quat).as_matrix()
        return matrix
    
    @property
    def left_finger(self):
        return self.manus_zmq_sub.left_joints
    
    @property
    def right_finger(self):
        return self.manus_zmq_sub.right_joints
    
    @property
    def left_short_skeleton(self):
        return self.manus_zmq_sub.left_short_skeleton
    
    @property
    def right_short_skeleton(self):
        return self.manus_zmq_sub.right_short_skeleton
    
    @property
    def left_full_skeleton(self):
        return self.manus_zmq_sub.left_full_skeleton
    
    @property
    def right_full_skeleton(self):
        return self.manus_zmq_sub.right_full_skeleton



LEFT_VIVE_TO_WRIST = np.array([
    [ 1, 0, 0],
    [ 0, -1, 0],
    [ 0, 0, -1]
])

RIGHT_VIVE_TO_WRIST = np.array([
    [ 1, 0, 0],
    [ 0, -1, 0],
    [ 0, 0, -1]
])

HEAD_VIVE_TO_CAMERA = np.array([
    [ -1, 0, 0],
    [ 0, 0, 1],
    [ 0, 1, 0]
])

class MocapTeleVisionWrapper:
    def __init__(self):
        self.tv = MocapData()
        time.sleep(1.0) #wait for sensor data stream
        self.waist_yaw_ori=0.0
        self.first_frame = True 

    def get_data(self):
        if not self.first_frame:
            if not self.tv.manus_zmq_sub.updated:
                return None
        self.first_frame = False
        T = np.array([[1, 0, 0], 
                      [0, 1, 0],
                      [0, 0, 1]])
        # --------------------------------wrist-------------------------------------
        left_hand_pose = self.tv.left_hand
        # print("left_hand_pose:", left_hand_pose)
        right_hand_pose = self.tv.right_hand
        head_matrix = np.copy(self.tv.head_matrix)

        # waist_angles = R.from_matrix(head_m atrix[:3,:3]).as_euler('xyz', degrees=False)
        # waist_yaw=waist_angles[1] #get from from test
        # waist_yaw_delta=waist_yaw-self.waist_yaw_ori
        # self.waist_yaw_ori=waist_yaw

        # unitree_left_wrist = left_hand_pose
        # unitree_right_wrist = right_hand_pose
        left_hand_pose[:3, :] = np.dot(T, left_hand_pose[:3, :])
        right_hand_pose[:3, :] = np.dot(T, right_hand_pose[:3, :])
        head_matrix[:3, :] = np.dot(T, head_matrix[:3, :])
        head_matrix[:3, :3] = np.dot(head_matrix[:3, :3], HEAD_VIVE_TO_CAMERA.T)
        left_hand_pose[:3, :3] = np.dot(left_hand_pose[:3, :3], LEFT_VIVE_TO_WRIST.T)
        right_hand_pose[:3, :3] = np.dot(right_hand_pose[:3, :3], RIGHT_VIVE_TO_WRIST.T)
        
        # # Transfer from WORLD to HEAD coordinate (translation only).
        # unitree_left_wrist = np.copy(left_hand_pose)
        # unitree_right_wrist = np.copy(right_hand_pose)
        # unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_matrix[0:3, 3]
        # unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_matrix[0:3, 3]

        # unitree_left_hand=self.tv.left_finger.copy()
        # unitree_right_hand=self.tv.right_finger.copy()

        # --------------------------------offset-------------------------------------

        # head_rmat = head_matrix[:3, :3]
        head_rmat = head_matrix
        # The origin of the coordinate for IK Solve is the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to check it.
        # The origin of the coordinate of unitree_left_wrist is HEAD. So it is necessary to translate the origin of unitree_left_wrist from HEAD to WAIST.
        # unitree_left_wrist[0, 3] +=0.05
        # unitree_right_wrist[0,3] +=0.05

        # unitree_left_wrist[1, 3] -=0.05
        # unitree_right_wrist[1,3] +=0.05

        # unitree_left_wrist[2, 3] +=0.35
        # unitree_right_wrist[2,3] +=0.35

        # unitree_left_hand_joints = np.deg2rad([3 * unitree_left_hand[0], 0.7 * unitree_left_hand[3], 1.15 * (unitree_left_hand[5]+unitree_left_hand[6]+unitree_left_hand[7])/3, 1.05 * (unitree_left_hand[9]+unitree_left_hand[10]+unitree_left_hand[11])/3, 1.25 * (unitree_left_hand[13]+unitree_left_hand[14]+unitree_left_hand[15])/3, 1.45 * (unitree_left_hand[17]+unitree_left_hand[18]+unitree_left_hand[19])/3])
        # unitree_right_hand_joints = np.deg2rad([3 * unitree_right_hand[0], 0.7 * unitree_right_hand[3], 1.15 * (unitree_right_hand[5]+unitree_right_hand[6]+unitree_right_hand[7])/3, 1.05 * (unitree_right_hand[9]+unitree_right_hand[10]+unitree_right_hand[11])/3, 1.25 * (unitree_right_hand[13]+unitree_right_hand[14]+unitree_right_hand[15])/3, 1.45 * (unitree_right_hand[17]+unitree_right_hand[18]+unitree_right_hand[19])/3])
        unitree_left_short_skeleton = self.tv.left_short_skeleton.copy()
        unitree_right_short_skeleton = self.tv.right_short_skeleton.copy()
        
        # waist yaw 
        head_pos = matrix_to_pose(head_rmat)
        head_p = np.array(head_pos, dtype=np.float32).tolist()  # 转换为 float32 再转 list

        left_pos = matrix_to_pose(left_hand_pose)
        left_p = np.array(left_pos, dtype=np.float32).tolist()  # 转换为 float32 再转 list

        right_pos = matrix_to_pose(right_hand_pose)
        right_p = np.array(right_pos, dtype=np.float32).tolist()  # 转换为 float32 再转 list

        tip_indices = [0, 2, 4, 6, 8]
        unitree_left_short_skeleton = np.array(unitree_left_short_skeleton, dtype=np.float32)[tip_indices]
        unitree_right_short_skeleton = np.array(unitree_right_short_skeleton, dtype=np.float32)[tip_indices]
        for i in range(5):
            unitree_left_short_skeleton[i][1] = -unitree_left_short_skeleton[i][1]
            unitree_right_short_skeleton[i][1] = -unitree_right_short_skeleton[i][1]
        unitree_left_short_skeleton = unitree_left_short_skeleton.reshape(-1).tolist()
        unitree_right_short_skeleton = unitree_right_short_skeleton.reshape(-1).tolist()

        unitree_left_full_skeleton = self.tv.left_full_skeleton.copy()
        unitree_right_full_skeleton = self.tv.right_full_skeleton.copy()
        unitree_left_full_skeleton = np.array(unitree_left_full_skeleton, dtype=np.float32)
        unitree_right_full_skeleton = np.array(unitree_right_full_skeleton, dtype=np.float32)
        for i in range(25):
            unitree_left_full_skeleton[i][1] = -unitree_left_full_skeleton[i][1]
            unitree_right_full_skeleton[i][1] = -unitree_right_full_skeleton[i][1]
        unitree_left_full_skeleton = unitree_left_full_skeleton.reshape(-1).tolist()
        unitree_right_full_skeleton = unitree_right_full_skeleton.reshape(-1).tolist()

        self.tv.manus_zmq_sub.updated = False
        return head_p, left_p, right_p, unitree_left_short_skeleton, unitree_right_short_skeleton, unitree_left_full_skeleton, unitree_right_full_skeleton


class MocapUpperBodyTeleVisionWrapper:
    
    def __init__(self):
        self.tv = MocapData()
        time.sleep(1.0) #wait for sensor data stream

        self.waist_angles_ori = R.from_matrix(self.tv.head_matrix[:3,:3]).as_euler('xyz', degrees=False)
        
        self.head_matrix = self.tv.head_matrix
        T = np.array([[0, 0, 1], 
                      [1, 0, 0],
                      [0, 1, 0]])
        self.head_matrix[:3, : ] = np.dot(T, self.head_matrix[:3, :])
        # print("head matrix:",self.head_matrix)

    def get_data(self):
        T = np.array([[0, 0, 1], 
                      [1, 0, 0],
                      [0, 1, 0]])
        # --------------------------------wrist-------------------------------------
        left_hand_pose = np.copy(self.tv.left_hand)
        right_hand_pose = np.copy(self.tv.right_hand)
        head_matrix = np.copy(self.tv.head_matrix)


        left_hand_pose[:3, :] = np.dot(T, left_hand_pose[:3, :])
        right_hand_pose[:3, :] = np.dot(T, right_hand_pose[:3, :])
        head_matrix[:3, : ] = np.dot(T, head_matrix[:3, :])

        waist_matrix = np.dot(self.head_matrix[:3, :3].T, head_matrix[:3, :3])
        waist_angles = R.from_matrix(waist_matrix[:3, :3]).as_euler('xyz', degrees=False)

        left_hand_pose[:3, :3] = np.dot(left_hand_pose[:3, :3], LEFT_VIVE_TO_WRIST.T)
        right_hand_pose[:3, :3] = np.dot(right_hand_pose[:3, :3], RIGHT_VIVE_TO_WRIST.T)
        
        # Transfer from  tracker WORLD to G1 HEAD coordinate (translation only).
        unitree_left_wrist = np.copy(left_hand_pose)
        unitree_right_wrist = np.copy(right_hand_pose)

        #change arm to fixed head position
        # unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_matrix[0:3, 3]
        # unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_matrix[0:3, 3]

        unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_matrix[0:3, 3]
        unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_matrix[0:3, 3]

        # unitree_left_hand=self.tv.left_finger.copy()
        # unitree_right_hand=self.tv.right_finger.copy()
        unitree_left_short_skeleton = self.tv.left_short_skeleton.copy()
        unitree_right_short_skeleton = self.tv.right_short_skeleton.copy()

        # --------------------------------offset-------------------------------------
        head_position = head_matrix[:3,3] - self.head_matrix[:3, 3]
        head_rmat = head_matrix[:3, :3]
        
        

        return head_position, head_rmat, unitree_left_wrist, unitree_right_wrist, waist_angles, unitree_left_short_skeleton, unitree_right_short_skeleton

# demo=MocapTeleVisionWrapper()
# # # # while(1):
# # # #     # print(demo.get_data())
# # # #     demo.get_data()
# # # #     continue
# # # start = time.time()
# # time.sleep(1)
# # # head_pose,left_wrist_pose,right_wrist_pose,left_finger_joints,right_finger_joints,left_full,right_full = demo.get_data()
# # # old_joints=right_finger_joints
# while(1):
#     # t1=time.time()
#     head_pose,left_wrist_pose,right_wrist_pose,left_finger_joints,right_finger_joints,left_full,right_full = demo.get_data()

#     ##according to trial, manus finger hz>50
#     # end = time.time()
#     # print("delata t for record: ",end-t1)
#     # print("frequency:", 1.0 / (end - start))
#     # start = time.time()
#     # print(head_pose) 
#     # print(len(left_full)) 
#     # print(right_wrist_pose) 
#     # print(left_wrist_pose) 
#     print(left_wrist_pose[0], left_wrist_pose[1], left_wrist_pose[2], left_finger_joints[0])
    
    
#     # print("delta right fingers: ",np.array(right_finger_joints)-np.array(old_joints))
#     # old_joints=right_finger_joints
#     # time.sleep(0.033)
#     time.sleep(0.05) ##