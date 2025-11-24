import argparse
import queue
import sys
import tempfile

import sounddevice as sd
import soundfile as sf


def int_or_str(text):
    """将输入转换为整数或字符串"""
    try:
        return int(text)
    except ValueError:
        return text


def main():
    parser = argparse.ArgumentParser(
        description="音频录制工具 - 将麦克风输入保存为WAV文件",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-l", "--list-devices", action="store_true", help="显示可用音频设备列表并退出"
    )
    parser.add_argument(
        "filename", nargs="?", metavar="FILENAME", help="输出文件名（默认：临时文件）"
    )
    parser.add_argument(
        "-d", "--device", type=int_or_str, help="选择输入设备（ID或名称）"
    )
    parser.add_argument(
        "-r", "--samplerate", type=int, help="采样率（默认：设备默认值）"
    )
    parser.add_argument(
        "-c", "--channels", type=int, default=1, help="声道数（默认：1）"
    )
    parser.add_argument(
        "-t", "--subtype", type=str, help='音频格式（如 "PCM_24"，默认：PCM_16）'
    )

    args = parser.parse_args()

    # 处理设备列表请求
    if args.list_devices:
        print(sd.query_devices())
        return

    # 获取设备信息
    device_info = sd.query_devices(args.device, "input")
    samplerate = args.samplerate or int(device_info["default_samplerate"])

    # 准备输出文件
    filename = args.filename or tempfile.mktemp(prefix="rec_", suffix=".wav", dir="")

    # 音频数据队列
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        """音频流回调函数"""
        if status:
            print(f"警告: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print("开始录音... (按 Ctrl+C 停止)")
    try:
        with (
            sf.SoundFile(
                filename,
                mode="x",
                samplerate=samplerate,
                channels=args.channels,
                subtype=args.subtype,
            ) as file,
            sd.InputStream(
                samplerate=samplerate,
                device=args.device,
                channels=args.channels,
                callback=audio_callback,
            ),
        ):
            while True:
                file.write(audio_queue.get())
    except KeyboardInterrupt:
        print(f'\n录音完成! 保存至: "{filename}"')
    except Exception as e:
        sys.exit(f"错误: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
