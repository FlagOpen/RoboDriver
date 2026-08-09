"""Reusable manual and random action disturbance triggers."""

from __future__ import annotations

import os
import queue
import random
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from typing import Any

import logging_mp


logger = logging_mp.get_logger(__name__)


@dataclass
class DisturbanceConfig:
    """Configure an action override triggered manually or at random."""

    enabled: bool = False
    manual_enabled: bool = True
    manual_key: str = "s"
    # Optional evdev device path (e.g. /dev/input/event5) for the manual key.
    # When empty, all accessible input devices that can emit manual_key are
    # auto-detected, so the key works even if the process runs in the
    # background (no terminal focus needed).
    manual_device: str = ""
    random_enabled: bool = False
    random_min_interval_s: float = 10.0
    random_max_interval_s: float = 30.0
    duration_s: float = 0.5
    cooldown_s: float = 0.3
    action_overrides: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.manual_key) != 1:
            raise ValueError("disturbance.manual_key must be exactly one character")
        if self.duration_s <= 0:
            raise ValueError("disturbance.duration_s must be greater than zero")
        if self.cooldown_s < 0:
            raise ValueError("disturbance.cooldown_s cannot be negative")
        if self.random_min_interval_s <= 0:
            raise ValueError(
                "disturbance.random_min_interval_s must be greater than zero"
            )
        if self.random_max_interval_s < self.random_min_interval_s:
            raise ValueError(
                "disturbance.random_max_interval_s must be greater than or equal "
                "to disturbance.random_min_interval_s"
            )


class ActionDisturbance:
    """Apply configured overrides to actions when a trigger fires."""

    def __init__(self, config: DisturbanceConfig):
        self.config = config
        self._triggers: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
        self._evdev_devices: list[Any] = []
        self._evdev_threads: list[threading.Thread] = []
        self._active_until = 0.0
        self._last_trigger_at = float("-inf")
        self._next_random_at: float | None = None
        self._missing_action_keys: set[str] = set()

    def start(self) -> None:
        if not self.config.enabled:
            return
        if not self.config.action_overrides:
            logger.warning("Disturbance is enabled but action_overrides is empty")

        now = time.monotonic()
        self._schedule_next_random(now)

        if self.config.manual_enabled:
            # Prefer evdev: works system-wide, even when this process runs in
            # the background or over SSH (no terminal focus needed).
            if not self._start_evdev_listener():
                if not sys.stdin.isatty():
                    logger.warning(
                        "Manual disturbance trigger is unavailable because evdev "
                        "is not usable and stdin is not a TTY"
                    )
                else:
                    logger.info(
                        "Falling back to the terminal keyboard listener; the "
                        "manual disturbance key only works while this terminal "
                        "is focused"
                    )
                    self._keyboard_thread = threading.Thread(
                        target=self._read_keyboard,
                        name="disturbance-keyboard",
                        daemon=True,
                    )
                    self._keyboard_thread.start()

        modes = []
        if self.config.manual_enabled:
            modes.append(f"key '{self.config.manual_key}'")
        if self.config.random_enabled:
            modes.append(
                "random "
                f"{self.config.random_min_interval_s:g}-"
                f"{self.config.random_max_interval_s:g}s"
            )
        logger.info(f"Disturbance trigger enabled ({', '.join(modes) or 'no trigger'})")

    def stop(self) -> None:
        self._stop_event.set()
        for device in self._evdev_devices:
            try:
                device.close()
            except OSError:
                pass
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=1.0)
        for thread in self._evdev_threads:
            thread.join(timeout=1.0)

    def trigger(self, source: str = "manual") -> None:
        """Queue a trigger. This is also useful for future hardware inputs."""
        if self.config.enabled:
            self._triggers.put(source)

    @property
    def is_active(self) -> bool:
        """Whether the disturbance override is currently active."""
        return self.config.enabled and time.monotonic() < self._active_until

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return an action with disturbance overrides applied when active."""
        if not self.config.enabled:
            return action

        now = time.monotonic()
        source = self._get_trigger_source(now)
        if source is not None and now - self._last_trigger_at >= self.config.cooldown_s:
            self._last_trigger_at = now
            self._active_until = max(self._active_until, now + self.config.duration_s)
            logger.warning(
                f"Disturbance triggered by {source}; overriding action for "
                f"{self.config.duration_s:g}s"
            )

        if not self.is_active:
            return action

        disturbed_action = action.copy()
        for key, value in self.config.action_overrides.items():
            if key in action:
                disturbed_action[key] = value
            elif key not in self._missing_action_keys:
                self._missing_action_keys.add(key)
                logger.warning(f"Disturbance action key is not present: {key}")
        return disturbed_action

    def _get_trigger_source(self, now: float) -> str | None:
        source = None
        while True:
            try:
                source = self._triggers.get_nowait()
            except queue.Empty:
                break

        if self._next_random_at is not None and now >= self._next_random_at:
            source = "random timer"
            self._schedule_next_random(now)
        return source

    def _schedule_next_random(self, now: float) -> None:
        if not self.config.random_enabled:
            self._next_random_at = None
            return
        delay = random.uniform(
            self.config.random_min_interval_s,
            self.config.random_max_interval_s,
        )
        self._next_random_at = now + delay

    def _start_evdev_listener(self) -> bool:
        """Listen for the manual key on evdev input devices, system-wide.

        Returns True when at least one device listener was started.
        """
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError:
            logger.info(
                "evdev is not installed; using the terminal keyboard listener "
                "for the manual disturbance trigger (pip install evdev)"
            )
            return False

        target_code = ecodes.ecodes.get(f"KEY_{self.config.manual_key.upper()}")
        if target_code is None:
            logger.warning(
                "Cannot map disturbance.manual_key "
                f"'{self.config.manual_key}' to an evdev keycode"
            )
            return False

        devices = self._open_evdev_devices(InputDevice, ecodes, list_devices, target_code)
        if not devices:
            logger.warning(
                "No accessible evdev device can emit key "
                f"'{self.config.manual_key}'. To listen system-wide, grant read "
                "access to /dev/input (e.g. `sudo usermod -aG input $USER` then "
                "re-login), or point --robot.disturbance.manual_device at the "
                "device shown by `ls /dev/input/by-id`"
            )
            return False

        for device in devices:
            thread = threading.Thread(
                target=self._read_evdev_device,
                args=(device, target_code),
                name=f"disturbance-evdev-{os.path.basename(device.path)}",
                daemon=True,
            )
            thread.start()
            self._evdev_devices.append(device)
            self._evdev_threads.append(thread)

        logger.info(
            f"Disturbance key '{self.config.manual_key}' is listened system-wide "
            f"on: {', '.join(device.path for device in devices)}"
        )
        return True

    def _open_evdev_devices(
        self, InputDevice: Any, ecodes: Any, list_devices: Any, target_code: int
    ) -> list[Any]:
        """Open the evdev devices that can emit the manual key."""
        if self.config.manual_device:
            paths = [self.config.manual_device]
        else:
            paths = list_devices()

        devices = []
        for path in paths:
            try:
                device = InputDevice(path)
            except OSError as exc:
                if self.config.manual_device:
                    logger.warning(
                        f"Cannot open disturbance.manual_device {path}: {exc}"
                    )
                    return []
                logger.debug(f"Skipping evdev device {path}: {exc}")
                continue

            # An explicit device is trusted as-is; auto-detected devices must
            # actually be able to emit the target key.
            if self.config.manual_device:
                devices.append(device)
                continue

            keys = device.capabilities().get(ecodes.EV_KEY, [])
            if target_code in keys:
                devices.append(device)
            else:
                device.close()
        return devices

    def _read_evdev_device(self, device: Any, target_code: int) -> None:
        from evdev import ecodes

        try:
            fd = device.fileno()
            while not self._stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                for event in device.read():
                    # value == 1: key press only, ignore repeat and release
                    if (
                        event.type == ecodes.EV_KEY
                        and event.value == 1
                        and event.code == target_code
                    ):
                        self.trigger(f"key '{self.config.manual_key}'")
        except OSError as exc:
            if not self._stop_event.is_set():
                logger.warning(
                    f"Disturbance evdev listener on {device.path} stopped: {exc}"
                )

    def _read_keyboard(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                key = os.read(fd, 1).decode(errors="ignore")
                if key.lower() == self.config.manual_key.lower():
                    self.trigger(f"key '{self.config.manual_key}'")
        except OSError as exc:
            if not self._stop_event.is_set():
                logger.warning(f"Manual disturbance keyboard listener stopped: {exc}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
