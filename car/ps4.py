import time
import asyncio
import evdev
from evdev import InputDevice, categorize, ecodes
from multiprocessing import Process, Manager

import socket
import json

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
                elif(event.code == 1):
                    data["ly"] = event.value
                elif(event.code == 2):
                    data["rx"] = event.value

    asyncio.ensure_future(update_inputs(dev, data))
    loop = asyncio.get_event_loop()
    loop.run_forever()


def read_controller_socket(conn_type="TCP", frequency=20, port=8080):
    if conn_type=="UDP":
        socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.bind(('', port))
    else:
        socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.bind(('', port))
        socket.listen()
        socket,addr = socket.accept()
        print(socket, addr)
    
    while True:
        start = time.time()
        print("receiving")
        data_raw,addr = socket.recvfrom(512)
        print("received")
        socket_data = json.loads(data_raw)
        for key, value in socket_data:
            d[key] = value
        # ensure data is more that 0.5 seconds old, reset to center
        if abs(d["timestamp"]-time.time()) > 0.5:
            d["ly"] = 128
            d["rx"] = 128
            # invalid data do not use for training
            d["timestamp"] = -1
            print("OLD DATA FROM SOCKET")
        else:
            # override the timestamp to current timestamp
            d["timestamp"] = time.time()
        while time.time()-start<(1.0/frequency):
            pass


class PS4Interface:
    def __init__(self, connection_type="bluetooth"):
        manager = Manager()
        self.data = manager.dict({
            "cross": 0,
            "square": 0,
            "triangle": 0,
            "circle": 0,
            "ly": 128,
            "rx": 128, 
            "timestamp":time.time()
        })

        if connection_type=="bluetooth":
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            path=None
            for device in devices:
                if("Wireless Controller" in device.name):
                    path = device.path
            dev = InputDevice(path)
            controller_process = Process(target=update_inputs, args=(dev, self.data))

        elif connection_type=="websocket_UDP":
            # SOCK_DGRAM defines a UDP connection
            socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        elif connection_type=="websocket_TCP":
            # SOCK_STREAM defines a TCP connection
            socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket.bind('',8080)
        
        controller_process.start()