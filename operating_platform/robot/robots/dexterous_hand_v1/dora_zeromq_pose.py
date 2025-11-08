import zmq  
import pyarrow as pa  
from dora import Node  
  
ipc_address = "ipc:///tmp/dr-robot-dexterous-hand-pose"  

context = zmq.Context()  
socket = context.socket(zmq.PAIR)  
socket.bind(ipc_address)
print(f"Socket bound to {ipc_address}", flush=True)  
socket.setsockopt(zmq.SNDHWM, 2000)  
socket.setsockopt(zmq.SNDBUF, 2**25)  
running_server = True

def main():  
    node = Node()  
    print(5)
    for event in node: 
        # print(6) 
        if event["type"] == "INPUT":  
            # print(7)
            event_id = event["id"]  
            buffer_bytes = event["value"].to_numpy().tobytes()
            # print(8)
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
