"""AstriBot T1 单环境（方案 A）自检：只读，不下发任何动作指令。

用法：
    python -m robodriver_robot_astribott1_aio_py.check_env
"""

import sys

import numpy as np


def _show_versions():
    print("=" * 60)
    print(f"python   : {sys.version.split()[0]}")
    print(f"numpy    : {np.__version__}")
    try:
        import cv2

        print(f"cv2      : {cv2.__version__}")
    except Exception as e:
        print(f"cv2      : 导入失败 {e}")


def _show_sdk():
    import astribot_sdk

    print(f"sdk      : astribot_sdk {getattr(astribot_sdk, '__version__', '?')}")
    from astribot_sdk import Astribot, AstribotCamera

    return Astribot, AstribotCamera


def _check_joints(Astribot):
    print("-" * 60)
    astribot = Astribot(node_name="robodriver_check_env")
    parts = ["arm_left", "arm_right", "effector_left", "effector_right", "head", "torso", "chassis"]
    names, ok = [], []
    for part in parts:
        try:
            names.append(getattr(astribot, f"{part}_name"))
            ok.append(part)
        except AttributeError:
            print(f"[跳过] 该 SDK 版本无部件 {part}")

    positions = astribot.get_current_joints_position(names=names)
    desired = astribot.get_desired_joints_position(names=names)

    print("部件自由度与当前关节角（rad）:")
    for part, pos, des in zip(ok, positions, desired):
        first = ", ".join(f"{v:+.3f}" for v in list(pos)[:3])
        print(f"  {part:<16} dof={len(pos):<3} 当前[{first}...] desired[0]={des[0]:+.3f}")

    bad = [p for p, pos in zip(ok, positions) if any(v != v for v in pos)]
    if bad:
        print(f"[失败] 部件 {bad} 返回 NaN，SDK 与当前 numpy 不兼容")
        return None
    return astribot


def _check_camera(AstribotCamera):
    print("-" * 60)
    cam = AstribotCamera()
    if not cam.activate_camera():
        print("[失败] activate_camera() 返回 False，确认机器人端 meta_camera 是否在运行")
        return None
    if not cam.wait_for_frame(timeout=10.0):
        print("[失败] 10 秒内没有相机数据")
        return None

    print(f"可用相机: {cam.camera_names()}")
    colors = cam.get_all_color()
    if not colors:
        print("[失败] get_all_color() 返回空")
        return None

    for name, frame in colors.items():
        arr = np.asarray(frame)
        print(f"  {name:<16} dtype={arr.dtype} shape={arr.shape} min={arr.min()} max={arr.max()}")
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            print(f"[警告] {name} 形状异常，RoboDriver 侧需要额外处理")
        if arr.max() == arr.min():
            print(f"[警告] {name} 整帧同值，可能是未就绪或解码错误")
    return cam


def main():
    _show_versions()
    try:
        Astribot, AstribotCamera = _show_sdk()
    except Exception as e:
        print(f"[失败] astribot_sdk 导入失败: {type(e).__name__}: {e}")
        print("       说明当前环境不适合方案 A，改用双环境方案（见 README「退路」）")
        return 1

    astribot = _check_joints(Astribot)
    cam = _check_camera(AstribotCamera)

    # 取过帧后必须显式关相机、且先于 Astribot，否则 SDK 在解释器退出时 SIGABRT
    if cam is not None:
        cam.deactivate_camera()
        cam.shutdown()
    if astribot is not None:
        astribot.shutdown()

    ok = astribot is not None and cam is not None
    print("=" * 60)
    print("结论: " + ("关节与图像均正常，方案 A（单环境）可用" if ok else "存在失败项，建议改用双环境方案 B"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
