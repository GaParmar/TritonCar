import time
import asyncio
import evdev
from evdev import InputDevice, categorize, ecodes
from multiprocessing import Process, Manager

def update_inputs(dev, data):
    async def update_inputs(dev, data):
        async for event in dev.async_read_loop():
            data["timestamp"] = time.time()
            if event.type == 1:
                if(event.code == 304):
                    data["square"] = event.value
                if(event.code == 306):
                    data["circle"] = event.value
                if(event.code == 307):
                    data["triangle"] = event.value
                if(event.code == 305):
                    data["cross"] = event.value

            # 0 to 255
            elif(event.type == 3):
                if(event.code == 0):
                    data["lx"] = event.value
                elif(event.code == 1):
                    data["ly"] = event.value
                elif(event.code == 2):
                    data["rx"] = event.value
                elif(event.code == 5):
                    data["ry"] = event.value

    asyncio.ensure_future(update_inputs(dev, data))
    loop = asyncio.get_event_loop()
    loop.run_forever()


class PS4Interface:
    def __init__(self):
        manager = Manager()
        self.data = manager.dict({
            "cross": 0,
            "square": 0,
            "triangle": 0,
            "circle": 0,
            "lx": 128,
            "ly": 128,
            "rx": 128,
            "ry": 128, 
            "timestamp":time.time()
        })

        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        path=None
        for device in devices:
            if("Wireless Controller" in device.name):
                path = device.path

        dev = InputDevice(path)

        socket_process = Process(target=update_inputs, args=(dev, self.data))
        socket_process.start()
