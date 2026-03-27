#!/usr/bin/env python3
"""Interactive servo test for Hitec HS-5085MG on GPIO18.

Servo specs (HS-5085MG):
  - Digital, metal gear, micro
  - Pulse width: 900-2100 µs (Hitec standard)
  - Torque: 4.3 kg-cm @ 6V
  - Speed: 0.13 sec/60° @ 6V

Controls:
  0-180  — set angle directly
  f      — fire cycle (90 deg -> wait 1s -> back to 0)
  s      — sweep 0 -> 180 -> 0
  q      — quit (returns servo to 0 and detaches)
"""

import time
import sys

GPIO_PIN = 18
STEP_DELAY = 0.02  # seconds between sweep steps

# Hitec HS-5085MG pulse range (seconds)
MIN_PULSE = 0.0009  # 900 µs
MAX_PULSE = 0.0021  # 2100 µs


def angle_to_value(angle: float) -> float:
    """Map 0-180 degrees to gpiozero's -1..+1 range."""
    return max(-1.0, min(1.0, (angle / 90.0) - 1.0))


def main():
    try:
        from gpiozero import Servo
        from gpiozero.pins.pigpio import PiGPIOFactory
    except ImportError:
        print("gpiozero not installed. Run: pip install gpiozero pigpio")
        sys.exit(1)

    servo_kwargs = dict(
        min_pulse_width=MIN_PULSE,
        max_pulse_width=MAX_PULSE,
    )

    try:
        factory = PiGPIOFactory()
        servo = Servo(GPIO_PIN, pin_factory=factory, **servo_kwargs)
        print(f"Using pigpio on GPIO{GPIO_PIN}")
    except Exception:
        servo = Servo(GPIO_PIN, **servo_kwargs)
        print(f"Using default pin factory on GPIO{GPIO_PIN}")

    print(f"Pulse range: {MIN_PULSE*1e6:.0f}-{MAX_PULSE*1e6:.0f} µs (HS-5085MG)")

    # Start at 0 degrees
    servo.value = angle_to_value(0)
    print("Servo at 0 deg")

    print("\nCommands: 0-180 (angle), f (fire), s (sweep), q (quit)")

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "q":
                break

            elif cmd == "f":
                print("FIRE -> 90 deg")
                servo.value = angle_to_value(90)
                time.sleep(1.0)
                print("REARM -> 0 deg")
                servo.value = angle_to_value(0)

            elif cmd == "s":
                print("Sweep 0 -> 180")
                for a in range(0, 181, 2):
                    servo.value = angle_to_value(a)
                    time.sleep(STEP_DELAY)
                print("Sweep 180 -> 0")
                for a in range(180, -1, -2):
                    servo.value = angle_to_value(a)
                    time.sleep(STEP_DELAY)
                print("Sweep done")

            else:
                try:
                    angle = float(cmd)
                    if 0 <= angle <= 180:
                        servo.value = angle_to_value(angle)
                        print(f"Servo -> {angle} deg")
                    else:
                        print("Angle must be 0-180")
                except ValueError:
                    print("Unknown command. Use 0-180, f, s, or q")

    except KeyboardInterrupt:
        print()

    servo.value = angle_to_value(0)
    time.sleep(0.3)
    servo.detach()
    print("Servo detached. Done.")


if __name__ == "__main__":
    main()
