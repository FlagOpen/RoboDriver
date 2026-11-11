"""Sharpa Wave 灵巧手只读自检：连接、读关节/触觉，不下发任何动作。

用法：
    python -m robodriver_robot_astribott1_aio_py.check_env_sharpa
    python -m robodriver_robot_astribott1_aio_py.check_env_sharpa --start --tactile

说明：
  * 默认只 connect + 读状态，不设置控制模式，也不会调用 set_joint_position。
  * --start 会调用 wave.start() 以启动状态/触觉流，仍然不会下发动作。
  * --tactile 会初始化触觉并尝试读取每根手指一帧，需要 SDK 未被 Pilot 占用。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .sharpa_hand import LEFT, RIGHT, VALID_SIDES, load_sharpa_sdk


def _wait_for_devices(sharpa: Any, manager: Any, timeout: float, sides: List[str]) -> List[Any]:
    deadline = time.monotonic() + timeout
    last_log = 0.0
    last_error: Optional[Exception] = None
    infos: List[Any] = []
    while time.monotonic() < deadline:
        try:
            infos = [
                info
                for info in manager.get_all_devices()
                if info.device_type == sharpa.DeviceType.HAND
            ]
        except Exception as exc:
            last_error = exc
            infos = []
        # 心跳异步到达：必须等所有需要的 side 都出现，不能看到一只就返回
        found = {_side_name(sharpa, info) for info in infos}
        missing = [side for side in sides if side not in found]
        if infos and not missing:
            return infos
        now = time.monotonic()
        if now - last_log >= 2.0:
            print(f"等待 Sharpa Wave 心跳（已发现: {sorted(found) or '无'}，缺少: {missing}）...")
            last_log = now
        time.sleep(0.2)
    if last_error is not None:
        print(f"[警告] 设备发现异常: {type(last_error).__name__}: {last_error}")
    return infos


def _side_name(sharpa: Any, info: Any) -> str:
    return LEFT if info.hand_side == sharpa.HandSide.LEFT else RIGHT


def _select_devices(
    sharpa: Any,
    infos: List[Any],
    side: str,
    left_sn: Optional[str],
    right_sn: Optional[str],
) -> Dict[str, Any]:
    wanted_sides = [LEFT, RIGHT] if side == "both" else [side]
    by_side: Dict[str, List[Any]] = {}
    for info in infos:
        by_side.setdefault(_side_name(sharpa, info), []).append(info)

    selected: Dict[str, Any] = {}
    for wanted in wanted_sides:
        sn = left_sn if wanted == LEFT else right_sn
        candidates = by_side.get(wanted, [])
        if sn:
            match = [info for info in infos if info.sn == sn]
            if not match:
                raise ValueError(f"未找到 {wanted} 手 SN={sn}")
            selected[wanted] = match[0]
        elif candidates:
            selected[wanted] = candidates[0]
        else:
            raise ValueError(f"未发现 {wanted} 手")
    return selected


def _print_content(content: Dict[str, Any], shapes: Dict[str, Any]) -> None:
    for name, data in content.items():
        label = "?" if name is None else str(name)
        arr = np.asarray(data)
        if arr.size == 0:
            print(f"    {label:<12} empty shape={shapes.get(name)}")
            continue
        print(
            f"    {label:<12} dtype={arr.dtype} shape={arr.shape} "
            f"min={arr.min():.4g} max={arr.max():.4g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sharpa Wave read-only check")
    parser.add_argument("--side", choices=[*VALID_SIDES, "both"], default="both")
    parser.add_argument("--left-sn", default=None)
    parser.add_argument("--right-sn", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--sdk-root", default=None, help="SDK 根目录，默认 /opt 或 ~/.local/opt")
    parser.add_argument("--start", action="store_true", help="调用 wave.start()，仍然不下发动作")
    parser.add_argument("--tactile", action="store_true", help="初始化触觉并读取每指一帧")
    args = parser.parse_args()

    if args.sdk_root:
        os.environ["SHARPA_SDK_ROOT"] = os.path.abspath(os.path.expanduser(args.sdk_root))

    print("=" * 64)
    try:
        sharpa = load_sharpa_sdk()
    except Exception as exc:
        print(f"[失败] Sharpa SDK 导入失败: {type(exc).__name__}: {exc}")
        print("       请安装 sharpa-wave-sdk deb，或设置 SHARPA_SDK_ROOT。")
        return 1

    print(f"python : {sys.version.split()[0]}")
    print(f"numpy  : {np.__version__}")
    print(f"sharpa : {sharpa.__file__}")

    try:
        manager = sharpa.SharpaWaveManager.get_instance()
    except Exception as exc:
        print(f"[失败] 初始化 SharpaWaveManager 失败: {type(exc).__name__}: {exc}")
        return 1

    sides = [LEFT, RIGHT] if args.side == "both" else [args.side]
    infos = _wait_for_devices(sharpa, manager, args.timeout, sides)
    if not infos:
        print(f"[失败] {args.timeout:.1f} 秒内未发现 Sharpa Wave")
        manager.disconnect_all()
        return 1

    print("-" * 64)
    for info in infos:
        print(
            f"发现: SN={info.sn} side={_side_name(sharpa, info)} ip={info.ip} "
            f"firmware={info.firmware_version} pn={info.pn} "
            f"tactile={info.has_fingertip_tactile()}"
        )
        print(f"      info: {info.to_json_string()}")

    try:
        selected = _select_devices(sharpa, infos, args.side, args.left_sn, args.right_sn)
    except ValueError as exc:
        print(f"[失败] {exc}")
        manager.disconnect_all()
        return 1

    need_start = args.start or args.tactile
    waves: Dict[str, Any] = {}
    started: List[str] = []
    try:
        for side, info in selected.items():
            print("-" * 64)
            print(f"连接 {side} 手: SN={info.sn}")
            wave = manager.connect(info.sn, not args.tactile)
            waves[side] = wave

            if need_start:
                if wave.start():
                    pass
                elif (
                    args.tactile
                    and wave.retry_tactile_alternate_port()
                    and wave.is_hand_ready()
                    and wave.is_tactile_ready()
                ):
                    print(f"{side} 手触觉端口冲突，已切换到备用端口")
                else:
                    print(f"[失败] {side} 手 start() 返回 False")
                    continue

                if args.tactile and not wave.is_tactile_ready():
                    if not (
                        wave.retry_tactile_alternate_port()
                        and wave.is_tactile_ready()
                    ):
                        print(f"[失败] {side} 手触觉未就绪")
                        continue
                    print(f"{side} 手触觉端口冲突，已切换到备用端口")

                started.append(side)
                print(f"{side} 手已启动（未下发动作）")

            states = wave.get_states()
            print(f"  state.angles: {len(states.angles)} 个")
            if states.angles:
                first = ", ".join(f"{v:+.3f}" for v in states.angles[:5])
                print(f"  前 5 个关节(rad): [{first}...]")

            error, positions = wave.get_joint_position_rad()
            if error.code == 0:
                print(f"  关节位置(rad): {len(positions)} 个")
                print("  " + ", ".join(f"{v:+.3f}" for v in positions))
            else:
                print(f"  [警告] get_joint_position_rad: {error.message}")

            tactile_api = getattr(wave, "has_tactile_support", None)
            tactile_supported = tactile_api() if callable(tactile_api) else info.has_fingertip_tactile()
            print(f"  tactile supported: {tactile_supported}")
            variant_api = getattr(wave, "get_hand_variant", None)
            print(f"  hand variant: {variant_api() if callable(variant_api) else info.pn}")
            if args.tactile:
                print(f"  tactile summary: {wave.tactile_summary()}")
                channels = range(5, 10) if side == LEFT else range(0, 5)
                for channel in channels:
                    frame = wave.fetch_tactile_frame(channel, 1.0)
                    if frame is None:
                        print(f"  tactile ch{channel}: 无数据（检查触觉端口/网线）")
                        continue
                    print(f"  tactile ch{channel}: ts={frame.get('ts')}")
                    _print_content(frame.get("content") or {}, frame.get("shape") or {})

        ok = all(side in started for side in selected) if need_start else bool(selected)
        print("=" * 64)
        print("结论: " + ("只读检查通过（未下发动作）" if ok else "存在失败项，请查看上面的日志"))
        return 0 if ok else 1
    finally:
        for side, wave in waves.items():
            try:
                if side in started:
                    wave.stop()
                wave.destroy()
            except Exception as exc:
                print(f"[警告] 关闭 {side} 手失败: {exc}")
        try:
            manager.disconnect_all()
        except Exception as exc:
            print(f"[警告] manager.disconnect_all 失败: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
