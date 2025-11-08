import zmq  
import pyarrow as pa  
from dora import Node  
  
ipc_address = "ipc:///tmp/dr-robot-dexterous-hand-image"  

context = zmq.Context()  
socket = context.socket(zmq.PAIR)  
socket.bind(ipc_address)
print(f"Socket bound to {ipc_address}", flush=True)  
socket.setsockopt(zmq.SNDHWM, 2000)  
socket.setsockopt(zmq.SNDBUF, 2*1024*1024)  
running_server = True

def main():  
    node = Node()  
    print(1)
    for event in node: 
        # print(2) 
        if event["type"] == "INPUT":  
            # print(3)
            event_id = event["id"]  
            buffer_bytes = event["value"].to_numpy().tobytes()
            # print(4)
            try:  
                socket.send_multipart([
                    event_id.encode('utf-8'),
                    buffer_bytes
                ], flags=zmq.NOBLOCK)
                # print("Send successful", flush=True) 
            except zmq.Again: 
                pass 
                print("[WARN] 发送缓冲区满，丢弃数据", flush=True)
            except Exception as e:  
                print(f"[ERROR] Send failed: {e}", flush=True)
                  
        elif event["type"] == "STOP":  
            break  
    
    running_server=False
    socket.close()  
    context.term()  
  
if __name__ == "__main__":  
    main()
