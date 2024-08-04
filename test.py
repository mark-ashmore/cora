import asyncio
from time import sleep

from agent_actions.kasa_control import turn_on_lamp, turn_off_lamp

asyncio.run(turn_on_lamp())
sleep(10)
asyncio.run(turn_off_lamp())
