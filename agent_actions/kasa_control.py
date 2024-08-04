import asyncio
import os
from kasa import Discover


async def turn_on_lamp():
    dev = await Discover.discover_single(os.environ["KASA_ADDRESS"])
    await dev.turn_on()
    await dev.update()


async def turn_off_lamp():
    dev = await Discover.discover_single(os.environ["KASA_ADDRESS"])
    await dev.turn_off()
    await dev.update()
