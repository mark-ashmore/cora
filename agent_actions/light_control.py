"""A class for Hue light controls and Kasa plug controls."""

import json
import os

from python_hue_v2 import Hue
from kasa import Discover

from utils import paths


class Lights:
    """Class for Hue light controls."""

    light_app_names = {
        "all": ["all"],
        "bridge": ["Bridge"],
        "engineering": ["Engineering"],
        "fan": ["Fan 1", "Fan 2"],
        "living room": ["Living room 1", "Living room 2"],
        "spyro": ["Spyro"],
        "kitchen": ["kitchen"],
    }

    def __init__(self) -> None:
        self.hue = Hue(os.environ["HUE_IP_ADDRESS"], os.environ["HUE_APPLICATION_KEY"])

    async def turn_on_kitchen(self):
        dev = await Discover.discover_single(os.environ["KASA_ADDRESS"])
        await dev.turn_on()
        await dev.update()

    async def turn_off_kitchen(self):
        dev = await Discover.discover_single(os.environ["KASA_ADDRESS"])
        await dev.turn_off()
        await dev.update()

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

    @staticmethod
    def get_light_value(entity_type: str, entity_span: str) -> str:
        """Get light value."""
        light_name_dict = json.load(
            (paths.ENTITIES_PATH / "light_name.json").open("r", encoding="utf-8")
        )
        labels = light_name_dict["labels"]
        for label in labels:
            if label["label"] == entity_type:
                custom_entities = label["custom_entities"]
        if custom_entities:
            for entity_value, entity_synonyms in custom_entities.items():
                if entity_span in entity_synonyms:
                    return entity_value
        return ""

    @staticmethod
    def get_light_names(entity_type: str, entity_span: str) -> list:
        if entity_span == "all":
            return [
                light for values in Lights.light_app_names.values() for light in values
            ]
        if entity_type and entity_span:
            light_value = Lights.get_light_value(entity_type, entity_span)
            if light_value:
                return Lights.light_app_names[light_value]
        return []
