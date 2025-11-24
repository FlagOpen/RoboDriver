import time

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf


def list_microphones():
    """列出所有可用的音频输入设备"""
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            input_devices.append((i, device["name"]))

    if not input_devices:
        print("❌ 未找到麦克风设备")
        return None

    print("\n可用的麦克风设备：")
    for idx, (id, name) in enumerate(input_devices):
        print(f"  {idx}. [{id}] {name}")

    return input_devices


def record_audio(duration=5, samplerate=44100, channels=1, device_id=None):
    """录制音频并保存为WAV文件"""
    print(f"\n🎙️ 准备录制 {duration} 秒音频...")
    print(f"  采样率: {samplerate} Hz | 通道: {channels} | 设备: {device_id}")

    # 开始录制
    print("🔴 正在录制... (请说话)")
    start_time = time.time()
    audio_data = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        device=device_id,
    )
    sd.wait()  # 等待录制完成
    record_time = time.time() - start_time
    print(f"✅ 录制完成! 实际录制时间: {record_time:.2f}秒")

    return audio_data, samplerate


def save_wav(audio_data, samplerate, filename="./cache/recorded_audio.wav"):
    """保存音频为WAV文件"""
    sf.write(filename, audio_data, samplerate)
    print(f"\n💾 音频已保存: '{filename}'")
    return filename


def play_audio(filename):
    """播放WAV文件"""
    print(f"\n🔊 播放音频: {filename}")
    data, fs = sf.read(filename)
    sd.play(data, fs)
    sd.wait()  # 等待播放完成
    print("✅ 播放结束")


def plot_waveform(filename):
    """绘制音频波形图（可选）"""
    try:
        data, fs = sf.read(filename)
        duration = len(data) / fs

        plt.figure(figsize=(10, 4))
        time = np.linspace(0, duration, len(data))

        if data.ndim == 1:  # 单声道
            plt.plot(time, data)
            plt.title("单声道音频波形")
        else:  # 立体声
            plt.plot(time, data[:, 0], label="左声道")
            if data.shape[1] > 1:
                plt.plot(time, data[:, 1], label="右声道", alpha=0.7)
            plt.title("立体声音频波形")
            plt.legend()

        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./cache/audio_waveform.png")
        print("\n📊 音频波形图已保存: './cache/audio_waveform.png'")
        plt.show()
    except Exception as e:
        print(f"⚠️ 绘制波形时出错: {e}")


def main():
    print("=" * 50)
    print("🎤 麦克风测试工具 (采集→保存WAV→播放)")
    print("=" * 50)

    # 1. 列出麦克风设备
    devices = list_microphones()
    if not devices:
        return

    # 2. 选择设备
    while True:
        try:
            choice = int(input("\n请选择麦克风设备编号: "))
            if 0 <= choice < len(devices):
                device_id = devices[choice][0]
                break
            print("❌ 无效的选择，请重试")
        except ValueError:
            print("❌ 请输入数字")

    # 3. 设置录制参数
    duration = float(input("请输入录制时长(秒) [默认5]: ") or 5)
    samplerate = int(input("请输入采样率(Hz) [默认44100]: ") or 44100)

    # 4. 录制音频
    audio_data, fs = record_audio(
        duration=duration,
        samplerate=samplerate,
        channels=1,  # 单声道录制（更通用）
        device_id=device_id,
    )

    # 5. 保存为WAV
    filename = save_wav(audio_data, fs)

    # 6. 播放测试
    play_audio(filename)

    # 7. 可选：显示波形
    if input("\n是否显示音频波形? (y/n) [n]: ").lower() == "y":
        plot_waveform(filename)

    print("\n" + "=" * 50)
    print("✅ 测试完成! 所有步骤成功执行")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        print("提示: 请确保麦克风已连接且被系统识别")
