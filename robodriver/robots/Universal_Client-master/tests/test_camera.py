#!/usr/bin/env python3
import cv2
print(111)
from pyorbbecsdk import ObCamera

def main():
    camera = None  # 先初始化相机变量，避免未定义报错
    try:
        # 1. 创建相机实例（包含在大的try块中，覆盖所有相机操作）
        print("1")
        camera = ObCamera()
        if camera is None:
            print("Failed to create ObCamera instance!")
            return

        # 2. 打开相机
        if not camera.open():
            print("Failed to open camera!")
            return

        print("Camera opened successfully! Press 'q' to quit.")

        # 3. 循环获取帧，添加退出条件和帧处理
        while True:
            # 获取帧并判断有效性
            frame = camera.get_frame()
            if frame is None or frame.empty():  # 判断帧是否有效（根据pyorbbecsdk实际情况调整）
                print("Failed to get valid frame, skipping...")
                continue  # 跳过无效帧，避免卡死

            # 显示帧（OpenCV窗口显示，必须步骤）
            cv2.imshow("Gemini355 Camera Frame", frame)

            # 监听键盘事件：按'q'退出循环（关键：解决无限死循环）
            # waitKey(1) 阻塞1毫秒，监听键盘输入，返回值为ASCII码
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit command received, exiting...")
                break  # 退出while循环

    except Exception as e:  # 正确拼写：except
        print(f"Error occurred: {e}")
    finally:
        # 确保相机正常关闭，窗口正常销毁
        if camera is not None:
            camera.close()
            print("Camera closed successfully!")
        cv2.destroyAllWindows()
        print("All OpenCV windows destroyed!")

if __name__ == "__main__":
    main()