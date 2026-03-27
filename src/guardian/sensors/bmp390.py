"""BMP390 barometer over I2C on Raspberry Pi."""

from guardian.sensors.base import BarometerABC


class BMP390Barometer(BarometerABC):
    """Real BMP390 barometer connected via I2C (pins 3-6 on RPi Zero 2W)."""

    def __init__(self):
        self._sensor = None
        self._reference_alt = 0.0

    def _ensure_init(self) -> None:
        if self._sensor is not None:
            return
        import board
        import busio
        import adafruit_bmp3xx

        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_bmp3xx.BMP3XX_I2C(i2c)
        self._sensor.pressure_oversampling = 8
        self._sensor.temperature_oversampling = 2

    def set_reference(self) -> None:
        self._ensure_init()
        self._reference_alt = self._sensor.altitude
        print(f"BMP390 reference altitude: {self._reference_alt:.1f}m")

    def read_altitude_m(self) -> float:
        self._ensure_init()
        return self._sensor.altitude

    def get_altitude_delta_m(self) -> float:
        return self.read_altitude_m() - self._reference_alt
