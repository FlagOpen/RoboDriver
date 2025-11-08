import zmq  
import pyarrow as pa  
from dora import Node  
  
ipc_address = "ipc:///tmp/dr-robot-dexterous-hand-plot"  

context = zmq.Context()  
socket = context.socket(zmq.PAIR)  
socket.bind(ipc_address)
print(f"Socket bound to {ipc_address}", flush=True)  
socket.setsockopt(zmq.SNDHWM, 2000)  
socket.setsockopt(zmq.SNDBUF, 2**25)  
running_server = True

def main():  
    node = Node()  
    print(9)
    for event in node: 
        # print(10) 
        if event["type"] == "INPUT":  
            # print(11)
            event_id = event["id"] 
            # print(event_id) 
            buffer_bytes = event["value"].to_numpy().tobytes()
            # print(len(buffer_bytes))
            # print(12)
            try:  
                socket.send_multipart([
                    event_id.encode('utf-8'),
                    buffer_bytes
                ], flags=zmq.NOBLOCK)
                # print("Send successful", flush=True) 
            except zmq.Again:  
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
