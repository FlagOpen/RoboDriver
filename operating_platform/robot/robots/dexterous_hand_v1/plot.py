import zmq
import matplotlib.pyplot as plt  
import numpy as np  
from mpl_toolkits.mplot3d import Axes3D  
from scipy.spatial.transform import Rotation as R  
import threading
import json
import time
plot_ipc_address = "ipc:///tmp/dr-robot-dexterous-hand-plot"   
plot_context = zmq.Context()  
plot_socket = plot_context.socket(zmq.PAIR)  
plot_socket.connect(plot_ipc_address)  
plot_socket.setsockopt(zmq.RCVTIMEO, 300)

recv_wrist_data = {}  
recv_finger_data = {}  
recv_head_data = {}  
recv_full_skeleton = {}  
lock = threading.Lock()  

def pose_recv_server():
    while True:    
        try:   
            
            message_parts = plot_socket.recv_multipart()  
            
            if len(message_parts) < 1:    
                continue    
    
            event_id = message_parts[0].decode('utf-8')  
            buffer_bytes = message_parts[1]  
            # print(event_id)
            # print(len(buffer_bytes))
            # print(1)
            # 检查是否有第三个部分（metadata）  
            metadata = {}  
            if len(message_parts) >= 3:  
                try:  
                    metadata = json.loads(message_parts[2].decode('utf-8'))  
                except:  
                    metadata = {}  
    
            if 'wrist_left' in event_id:    
                pose_left_data = np.frombuffer(buffer_bytes, dtype=np.float32) 
                # print(pose_left_data)
                with lock:    
                    recv_wrist_data[event_id] = pose_left_data
 
            elif 'wrist_right' in event_id:    
                pose_right_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
                with lock:    
                    recv_wrist_data[event_id] = pose_right_data   
    
            elif 'finger_left' in event_id:    
                finger_left_data = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_finger_data[event_id] = finger_left_data  
            elif 'finger_right' in event_id:    
                finger_right_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
                with lock:    
                    recv_finger_data[event_id] = finger_right_data   
    
            elif 'head' in event_id:    
                head_data = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_head_data[event_id] = head_data

            elif 'left_full_skeleton' in event_id:    
                left_full_skeleton = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_full_skeleton[event_id] = left_full_skeleton
            
            elif 'right_full_skeleton' in event_id:    
                right_full_skeleton = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_full_skeleton[event_id] = right_full_skeleton

        except zmq.Again:    
            continue    
        except Exception as e:    
            print("recv error:", e)    
            break


class DexterousHandVisualizer:  
    def __init__(self):  
        self.fig = plt.figure(figsize=(10, 7), dpi=100)  
        self.ax = self.fig.add_subplot(111, projection='3d')  
        self.setup_plot()  
          
    def setup_plot(self):  
        self.ax.set_xlabel('X')  
        self.ax.set_ylabel('Y')   
        self.ax.set_zlabel('Z')  
        self.ax.set_title("Dexterous Hand Pose Visualization")  
        self.ax.set_xlim(-0.5, 0.5)  # 添加默认视图范围
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)
          
    def update_poses(self, recv_wrist_data, recv_head_data, recv_full_skeleton):  
        self.ax.cla()  
        self.setup_plot()  
        
        
        left_wrist_pos = np.zeros(3)
        right_wrist_pos = np.zeros(3)
        left_wrist_rot = np.zeros(4)
        right_wrist_rot = np.zeros(4)

        # 可视化左右手腕位置  
        if recv_wrist_data and 'wrist_left' in recv_wrist_data and len(recv_wrist_data['wrist_left']) >= 7:  
            pos = np.array(recv_wrist_data['wrist_left'][:3])  # 位置 
            rot = recv_wrist_data['wrist_left'][3:7]  # 四元数  
            left_wrist_pos = pos.copy()
            left_wrist_rot = rot.copy()
            self.plot_pose(pos, rot, 'cyan', 'Left Wrist')  
              
        if recv_wrist_data and 'wrist_right' in recv_wrist_data and len(recv_wrist_data['wrist_right']) >= 7:  
            pos = np.array(recv_wrist_data['wrist_right'][:3])  
            rot = recv_wrist_data['wrist_right'][3:7]
            right_wrist_pos = pos.copy()  
            right_wrist_rot = rot.copy()
            self.plot_pose(pos, rot, 'magenta', 'Right Wrist')  
              
        # 可视化头部姿态  
        if recv_head_data and 'head_pose' in recv_head_data and len(recv_head_data['head_pose']) >= 7:  
            pos = np.array(recv_head_data['head_pose'][:3])  
            rot = recv_head_data['head_pose'][3:7]  
            self.plot_pose(pos, rot, 'green', 'Head')  

        # 可视化手指
        if recv_full_skeleton:
            if 'left_full_skeleton' in recv_full_skeleton and len(recv_full_skeleton['left_full_skeleton']) >= 75:  
                skeleton_data = recv_full_skeleton['left_full_skeleton'][:75].reshape(25, 3)  
                self.plot_full_skeleton(  
                    skeleton_data,   
                    left_wrist_pos,  
                    left_wrist_rot,  
                    'cyan',   
                    'Left Hand Skeleton'  
                )  

            if 'right_full_skeleton' in recv_full_skeleton and len(recv_full_skeleton['right_full_skeleton']) >= 75: 
                skeleton_data = recv_full_skeleton['right_full_skeleton'][:75].reshape(25, 3)  
                self.plot_full_skeleton(  
                    skeleton_data,  
                    right_wrist_pos,   
                    right_wrist_rot,  
                    'magenta',   
                    'Right Hand Skeleton'  
                )

        plt.draw()  
        plt.pause(0.01)  
          
    def plot_pose(self, position, quaternion, color, label):  
        # 绘制坐标系  
        origin = np.array(position)
        try:
            rot_matrix = R.from_quat(quaternion).as_matrix()  
        except:
            rot_matrix = np.eye(3)  # 如果四元数无效，使用单位矩阵
          
        axis_length = 0.1  
        x_axis = origin + rot_matrix[:, 0] * axis_length  
        y_axis = origin + rot_matrix[:, 1] * axis_length    
        z_axis = origin + rot_matrix[:, 2] * axis_length  
          
        self.ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
                    c='red', linewidth=2, alpha=0.8)  # X轴
        self.ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
                    c='green', linewidth=2, alpha=0.8)  # Y轴
        self.ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
                    c='blue', linewidth=2, alpha=0.8)  # Z轴
        
        self.ax.text(origin[0], origin[1], origin[2], label, color='black', fontsize=8)

    
    def plot_full_skeleton(self, skeleton_points, wrist_pos, wrist_rot, color, label):     

        rot = R.from_quat(wrist_rot)    
        rot_matrix = rot.as_matrix()    
        

        wrist_pos = np.array(wrist_pos)    
        rotated_points = np.dot(skeleton_points, rot_matrix.T)    
        absolute_points = rotated_points + np.array(wrist_pos)  
        
        for point in absolute_points:  
            self.ax.scatter(point[0], point[1], point[2],     
                        c=color, s=30, alpha=0.8, marker='o')   

def main():  
    # 启动数据接收线程  
    recv_thread = threading.Thread(target=pose_recv_server, daemon=True)  
    recv_thread.start()  
      
    # 创建可视化器  
    visualizer = DexterousHandVisualizer()  
    plt.ion()  
    plt.show()   
    # 主循环  
    try:  
        while True:  
            with lock:  
                # 检查数据是否可用 
                # print(f"Data status - Wrist: {bool(recv_wrist_data)}, Head: {bool(recv_head_data)}, Skeleton: {bool(recv_full_skeleton)}") 
                if recv_wrist_data and recv_head_data and recv_full_skeleton:  
                    visualizer.update_poses(recv_wrist_data, recv_head_data, recv_full_skeleton)    
            plt.pause(0.033)    
    except KeyboardInterrupt:  
        print("可视化已停止")  
    finally: 
        plt.ioff() 
        plt.close(visualizer.fig)  
  
if __name__ == "__main__":  
    main()       