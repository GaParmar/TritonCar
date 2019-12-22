import os
import sys
import time
import json
import pdb

from motor import *
from ps4 import *
from utils import *

if __name__ == "__main__":
    # load parameters from config.json file
    with open('config.json') as json_file:
        config = json.load(json_file)

    ps4 = PS4Interface()
    camera = cv2.VideoCapture(config["camera_id"])
    motor = ArduinoMotor()
    log_buffer = []

    # ensure that log dir does not exist
    if not os.path.exists(config["log_path"])
        os.makedirs(config["log_path"])
        log_counter = 0
    else:
        # find the file index to continue from
        for f in os.listdir(config["log_path"]):
            val = int(f.split("_")[1].split(".")[0])
            log_counter = max(val+1, log_file_counter)

    pdb.set_trace()

    # start in manual mode
    state = "manual"
    logging = False

    # main control loop
    while True:
        ts_start = time.time()

        status, img = camera.read()
        assert status

        curr_data = {
            "throttle" : -1,
            "steer"    : -1,
            "timestamp": -1,
            "image"    : img
        }

        logging = False if ps4.data["square"] else logging
        state = "manual" if ps4.data["cross"] else state
        state = "autonomous" if ps4.data["circle"] else state
        logging = True if ps4.data["triangle"] else logging

        if state == "manual":
            # parse the ps4 data
            # ensure that data is not very stale
            if ps4.data["timestamp"]-time.time() > config["tolerance_time"]:
                curr_data["throttle"] = 90
                curr_data["steer"] = 90
            else:
                curr_data["throttle"] = ((ps4.data["ly"]-128)/128)*config["throttle_allowance"] + 90
                curr_data["steer"] = ((ps4.data["rx"]-128)/128)*config["steer_allowance"] + 90
            
                if logging:
                    log_buffer.append(curr_data)
                    if len(log_buffer)==config["samples_per_file"]:
                        p = Process(target=save_to_file, args=(log_dir, log_counter, deepcopy(log_buffer)))
                        p.start()
                        log_buffer = []
                        log_file_counter += 1

        elif state == "autonomous":
            curr_data["throttle"] = 90
            curr_data["steer"] = 90

        else:
            raise ValueError(f"state {state} not implemented yet")

        # send control to the motors
        motor.send_data(curr_data)

        if time.time() > (ts_start+config["refresh_time"]):
            print(f"main loop takes {time.time()-ts_start}")
        else:
            # busy wait while time requirement is met
            while time.time() < (ts_start+config["refresh_time"]):
                pass


