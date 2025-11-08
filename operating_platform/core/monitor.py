import threading
import requests
import json
import time

from operating_platform.robot.daemon import Daemon


class Monitor:
    def __init__(self, daemon: Daemon):
        self.daemon = daemon

        self.url = "http://localhost:8088/robot/update_machine_information"
        self.headers = {"Content-Type": "application/json"}

        self._lock = threading.Lock()  # 线程锁
        self._thread = threading.Thread(
            target=self.send_machine_info_periodically,
            daemon=True
        )

        self._running = False  # 控制线程启停

    def send_machine_info_periodically(self, interval_seconds=60):
        """定时发送设备信息（线程安全）"""
        while self._running:
            try:
                with self._lock:
                    if self.daemon.robot.status is None:
                        continue
                    json_data = self.daemon.get_status()
                    # json_data = json.dumps(self._machine_info_dict, ensure_ascii=False)
                
                response = requests.post(
                    self.url,
                    data=json_data,
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"[{time.ctime()}] 信息发送成功: {response.text}")
                else:
                    print(f"[{time.ctime()}] 发送失败，状态码: {response.status_code}")
            
            except requests.exceptions.RequestException as e:
                print(f"[{time.ctime()}] 请求异常: {str(e)}")
            
            time.sleep(interval_seconds)

    def start(self):
        """启动后台线程"""
        self._thread.start()
        self._running = True

    def stop(self):
        """停止后台线程"""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join()