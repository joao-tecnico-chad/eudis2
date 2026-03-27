#!/usr/bin/env python3
"""Interactive servo test for Hitec HS-5085MG on GPIO18.

Servo specs (HS-5085MG):
  - Digital, metal gear, micro
  - Pulse width: 900-2100 µs (Hitec standard)
  - Torque: 4.3 kg-cm @ 6V
  - Speed: 0.13 sec/60° @ 6V

Controls:
  0-180    — set angle directly (instant jump)
  f        — fire cycle (90 deg -> wait 1s -> back to 0)
  s        — sweep 0 -> 180 -> 0
  speed N  — set speed 1-100 (1=slowest, 100=instant). Default: 50
  q        — quit (returns servo to 0 and detaches)
"""

import time
import sys

GPIO_PIN = 18

# Hitec HS-5085MG pulse range (seconds)
MIN_PULSE = 0.0009  # 900 µs
MAX_PULSE = 0.0021  # 2100 µs

# Speed: 1 (slowest) to 100 (instant)
current_speed = 50
current_angle = 0.0


def angle_to_value(angle: float) -> float:
    """Map 0-180 degrees to gpiozero's -1..+1 range."""
    return max(-1.0, min(1.0, (angle / 90.0) - 1.0))


def move_to(servo, target_angle: float) -> None:
    """Move servo to target angle at the current speed."""
    global current_angle

    if current_speed >= 100:
        # Instant jump
        servo.value = angle_to_value(target_angle)
        current_angle = target_angle
        return

    # Calculate step size and delay from speed
    # speed 1 = 0.5 deg steps, 30ms delay (very slow)
    # speed 50 = 2 deg steps, 10ms delay
    # speed 99 = 5 deg steps, 2ms delay
    step_size = 0.5 + (current_speed / 100.0) * 4.5  # 0.5 - 5.0 deg
    step_delay = 0.030 - (current_speed / 100.0) * 0.028  # 30ms - 2ms

    direction = 1 if target_angle > current_angle else -1
    pos = current_angle

    while True:
        remaining = abs(target_angle - pos)
        if remaining < step_size:
            pos = target_angle
            servo.value = angle_to_value(pos)
            break
        pos += direction * step_size
        servo.value = angle_to_value(pos)
        time.sleep(step_delay)

    current_angle = target_angle


def main():
    global current_speed, current_angle

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
    current_angle = 0.0
    print("Servo at 0 deg")
    print(f"Speed: {current_speed}")

    print("\nCommands: 0-180 (angle), f (fire), s (sweep), speed N (1-100), q (quit)")

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "q":
                break

            elif cmd == "f":
                print("FIRE -> 90 deg")
                move_to(servo, 90)
                time.sleep(1.0)
                print("REARM -> 0 deg")
                move_to(servo, 0)

            elif cmd == "s":
                print("Sweep 0 -> 180")
                move_to(servo, 180)
                print("Sweep 180 -> 0")
                move_to(servo, 0)
                print("Sweep done")

            elif cmd.startswith("speed"):
                parts = cmd.split()
                if len(parts) == 2:
                    try:
                        val = int(parts[1])
                        if 1 <= val <= 100:
                            current_speed = val
                            print(f"Speed -> {current_speed}")
                        else:
                            print("Speed must be 1-100")
                    except ValueError:
                        print("Usage: speed N (1-100)")
                else:
                    print(f"Current speed: {current_speed}. Usage: speed N (1-100)")

            else:
                try:
                    angle = float(cmd)
                    if 0 <= angle <= 180:
                        move_to(servo, angle)
                        print(f"Servo -> {angle} deg")
                    else:
                        print("Angle must be 0-180")
                except ValueError:
                    print("Unknown command. Use 0-180, f, s, speed N, or q")

    except KeyboardInterrupt:
        print()

    move_to(servo, 0)
    time.sleep(0.3)
    servo.detach()
    print("Servo detached. Done.")


if __name__ == "__main__":
    main()
