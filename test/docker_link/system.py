import time

import docker

client = docker.from_env()

# ===== 1. 创建 IPC 管理容器 =====
print("启动 IPC 管理容器...")
ipc_manager = client.containers.run(
    "python:3.9",
    name="ipc-manager",
    command="""python -c '
import time;
from multiprocessing import shared_memory;
print("IPC 管理器: 初始化共享内存...");
shm = shared_memory.SharedMemory(create=True, size=100, name="my_region");
print("IPC 管理器: 共享内存已创建");
# 保持运行以便其他容器能访问
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    shm.close()
    shm.unlink()
    print("IPC 管理器: 已清理共享内存")'""",
    detach=True,
    ipc_mode="shareable",
)

# 等待容器就绪
start_time = time.time()
while time.time() - start_time < 15:
    ipc_manager.reload()
    if ipc_manager.status == "running":
        print("IPC 管理器已就绪")
        break
    elif ipc_manager.status == "exited":
        logs = ipc_manager.logs().decode("utf-8", errors="replace")
        raise RuntimeError(f"IPC 管理器意外退出! 日志:\n{logs}")
    time.sleep(0.5)
else:
    raise TimeoutError("IPC 管理器启动超时")

time.sleep(2)  # 确保共享内存已准备好

# ===== 2. 创建生产者容器 =====
producer = client.containers.run(
    "python:3.9",
    name="shm-producer",
    command="""python -c '
from multiprocessing import shared_memory;
print("生产者: 尝试连接共享内存...");
shm = shared_memory.SharedMemory(name="my_region");
shm.buf[:17] = b"Hello from Python";
print("生产者: 数据写入完成");
shm.close();'""",
    detach=True,
    ipc_mode="container:ipc-manager",
)

# ===== 3. 创建消费者容器 =====
consumer = client.containers.run(
    "python:3.9",
    name="shm-consumer",
    command="""python -c '
from multiprocessing import shared_memory;
print("消费者: 尝试连接共享内存...");
shm = shared_memory.SharedMemory(name="my_region");
data = bytes(shm.buf[:17]).decode();
print(f"消费者: 读取成功 → \'{data}\'");
shm.close();'""",
    detach=True,
    ipc_mode="container:ipc-manager",
)

# ===== 4. 等待执行完成并输出日志 =====
print("\n=== 执行监控 ===")
try:
    # 等待消费者完成
    for _ in range(15):
        consumer.reload()
        if consumer.status == "exited":
            break
        time.sleep(1)

    print("\n=== 最终结果 ===")
    print("消费者输出:")
    print(consumer.logs().decode("utf-8").strip())

    print("\n生产者输出:")
    print(producer.logs().decode("utf-8").strip())

finally:
    try:
        producer.stop(timeout=2)
        consumer.stop(timeout=2)
        ipc_manager.stop(timeout=2)
    except:
        pass
    try:
        producer.remove()
        consumer.remove()
        ipc_manager.remove()
    except:
        pass
