import argparse
import queue
import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation


def int_or_str(text):
    """Convert to int if possible, otherwise return string."""
    try:
        return int(text)
    except ValueError:
        return text


class AudioVisualizer:
    def __init__(self, args):
        self.args = args
        self.mapping = [c - 1 for c in args.channels]
        self.q = queue.Queue()
        self.setup_audio()
        self.setup_plot()

    def setup_audio(self):
        if self.args.samplerate is None:
            self.args.samplerate = sd.query_devices(self.args.device, "input")[
                "default_samplerate"
            ]

        self.length = int(
            self.args.window * self.args.samplerate / (1000 * self.args.downsample)
        )
        self.plotdata = np.zeros((self.length, len(self.args.channels)))

        self.stream = sd.InputStream(
            device=self.args.device,
            channels=max(self.args.channels),
            samplerate=self.args.samplerate,
            blocksize=self.args.blocksize,
            callback=self.audio_callback,
        )

    def audio_callback(self, indata, frames, time, status):
        """Audio data callback."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata[:: self.args.downsample, self.mapping])

    def setup_plot(self):
        """Initialize the plot."""
        plt.style.use("fast")
        self.fig, self.ax = plt.subplots()

        self.lines = self.ax.plot(self.plotdata)
        if len(self.args.channels) > 1:
            self.ax.legend([f"ch{c}" for c in self.args.channels], loc="upper right")

        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.length)
        self.ax.grid(axis="y", linestyle="--", alpha=0.7)
        self.ax.set_xticks([])
        self.fig.tight_layout(pad=0.5)

    def update_plot(self, frame):
        """Update plot with new audio data."""
        while not self.q.empty():
            data = self.q.get_nowait()
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:] = data

        for i, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, i])
        return self.lines


def main():
    parser = argparse.ArgumentParser(description="Real-time audio visualization")
    parser.add_argument(
        "-l", "--list-devices", action="store_true", help="list audio devices and exit"
    )
    parser.add_argument(
        "channels", type=int, default=[1], nargs="*", help="input channels (default: 1)"
    )
    parser.add_argument("-d", "--device", type=int_or_str, help="input device")
    parser.add_argument(
        "-w", "--window", type=float, default=200, help="window size (ms)"
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=30, help="update interval (ms)"
    )
    parser.add_argument("-b", "--blocksize", type=int, help="block size (samples)")
    parser.add_argument("-r", "--samplerate", type=float, help="sampling rate")
    parser.add_argument(
        "-n", "--downsample", type=int, default=10, help="downsampling factor"
    )

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    if any(c < 1 for c in args.channels):
        parser.error("Channel numbers must be >= 1")

    try:
        visualizer = AudioVisualizer(args)
        with visualizer.stream:
            ani = FuncAnimation(
                visualizer.fig,
                visualizer.update_plot,
                interval=args.interval,
                blit=True,
            )
            plt.show()
    except Exception as e:
        sys.exit(f"Error: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
