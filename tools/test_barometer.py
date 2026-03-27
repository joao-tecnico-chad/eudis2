#!/usr/bin/env python3
"""Interactive barometer test for BMP390 on I2C (SCL=GPIO3, SDA=GPIO2).

Uses max oversampling + moving average to reduce noise to ~1cm.

Controls:
  r      — set current altitude as reference (zero point)
  q      — quit

Continuously prints pressure, temperature, and altitude.
"""

import sys
import time
import select
from collections import deque

WINDOW_SIZE = 20  # moving average window


def main():
    try:
        import board
        import busio
        import adafruit_bmp3xx
    except ImportError:
        print("Missing deps. Run: pip install adafruit-circuitpython-bmp3xx")
        sys.exit(1)

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bmp3xx.BMP3XX_I2C(i2c)

    # Max oversampling for lowest noise
    sensor.pressure_oversampling = 32
    sensor.temperature_oversampling = 2

    # IIR filter coefficient (0, 2, 4, 8, 16, 32, 64, 128)
    sensor.filter_coefficient = 16

    print("BMP390 initialized on I2C (SCL=GPIO3, SDA=GPIO2)")
    print(f"Oversampling: pressure=32x, IIR filter=16")
    print(f"Moving average window: {WINDOW_SIZE} samples")
    print("Note: absolute altitude is approximate — use delta for relative changes")
    print("\nCommands: r (set reference), q (quit)")

    # Fill the buffer before starting
    print("Warming up...", end="", flush=True)
    alt_buffer = deque(maxlen=WINDOW_SIZE)
    for _ in range(WINDOW_SIZE):
        alt_buffer.append(sensor.altitude)
        time.sleep(0.05)
    print(" done.\n")

    reference_alt = 0.0

    try:
        while True:
            pressure = sensor.pressure
            temp = sensor.temperature
            alt_buffer.append(sensor.altitude)
            alt = sum(alt_buffer) / len(alt_buffer)
            delta = alt - reference_alt

            print(
                f"  P: {pressure:7.2f} hPa  |  T: {temp:5.1f} °C  |  "
                f"Alt: {alt:7.2f} m  |  Delta: {delta:+.2f} m",
                end="\r",
            )

            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "q":
                    break
                elif cmd == "r":
                    reference_alt = alt
                    print(f"\n  Reference set: {reference_alt:.2f} m")
                    print()

    except KeyboardInterrupt:
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
