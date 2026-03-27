#!/usr/bin/env python3
"""Interactive barometer test for BMP390 on I2C (SCL=GPIO3, SDA=GPIO2).

Controls:
  r      — set current altitude as reference (zero point)
  q      — quit

Continuously prints pressure, temperature, and altitude.
"""

import sys
import time
import select


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
    sensor.pressure_oversampling = 8
    sensor.temperature_oversampling = 2

    print("BMP390 initialized on I2C (SCL=GPIO3, SDA=GPIO2)")
    print(f"Sea level pressure: {sensor.sea_level_pressure:.1f} hPa")
    print("\nCommands: r (set reference), q (quit)")
    print("Reading every 0.5s...\n")

    reference_alt = 0.0

    try:
        while True:
            pressure = sensor.pressure
            temp = sensor.temperature
            alt = sensor.altitude
            delta = alt - reference_alt

            print(
                f"  P: {pressure:7.2f} hPa  |  T: {temp:5.1f} °C  |  "
                f"Alt: {alt:7.2f} m  |  Delta: {delta:+.2f} m",
                end="\r",
            )

            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0.5)[0]:
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
