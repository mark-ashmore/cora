import time
from python_hue_v2 import Hue, BridgeFinder

HUE = Hue(
    "0017887e2e47.local.", "YCMWZq3gYGb7sM8f4-m3POpVnHvneg8VPh96EzYV"
)  # create Hue instance


class Lights:
    def __init__(self, hue: Hue) -> None:
        """Class for Hue light controls."""
        self.hue = hue

    def turn_on_light(self, name: str, brightness: float = 100.00) -> bool:
        """Turn on a Hue light by its name."""
        lights = self.hue.lights
        for light in lights:
            metadata = light.metadata
            if metadata["name"] == name:
                light.on = True
                light.brightness = brightness
                return True
        return False

    def turn_off_light(self, name: str) -> bool:
        """Turn off a Hue light by its name."""
        lights = self.hue.lights
        for light in lights:
            metadata = light.metadata
            if metadata["name"] == name:
                light.on = False
                return True
        return False

    def change_light_brightness(self, name: str, brightness: float) -> bool:
        """Change a Hue light brightness by its name."""
        lights = self.hue.lights
        for light in lights:
            metadata = light.metadata
            if metadata["name"] == name:
                light.brightness = brightness
                return True
        return False


if __name__ == "__main__":
    my_lights = Lights(HUE)

    # my_lights.turn_off_light('Lounge lamp')
    my_lights.turn_on_light("Living room 1", 100.00)
    time.sleep(5)
    # my_lights.turn_on_light('Lounge lamp')
    my_lights.change_light_brightness("Living room 1", 0.00)
    my_lights.turn_off_light("Living room 1")
