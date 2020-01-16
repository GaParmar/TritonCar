import os, sys, time
import json, pdb
import cv2
from copy import deepcopy

from PIL import Image
import numpy as np

from motor import *
from ps4 import *
from utils import *

from server.network import *

def norm_split(img):
    left = np.array(img.crop((0,0,320,240)).resize([160,120]))
    right = np.array(img.crop((320,0,640,240)).resize([160,120]))
    img = np.concatenate((left,right), axis=2).astype(np.float32)
    # normalize to range [0,1] from [0,255]
    # img /= 255.0
    return img

if __name__ == "__main__":
    # load parameters from config.json file
    with open('car/config.json') as json_file:
        config = json.load(json_file)
    
    # load trained model weights
    model_path = config["trained_model_path"]
    auto_model = None
    if model_path is not None:
        auto_model = KerasLinearPolicy()
        auto_model.model.load_weights(model_path)

    ps4 = PS4Interface(connection_type=config["controller_type"])
    camera = cv2.VideoCapture(config["camera_id"])
    motor = ArduinoMotor()
    log_buffer = []
    log_counter=0

    # ensure that log dir does not exist
    if not os.path.exists(config["log_path"]):
        os.makedirs(config["log_path"])
        log_counter = 0
    else:
        # find the file index to continue from
        for f in os.listdir(config["log_path"]):
            val = int(f.split("_")[1].split(".")[0])
            log_counter = max(val+1, log_counter)

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
        oldstate = state
        state = "manual" if ps4.data["cross"] else state
        state = "autonomous" if ps4.data["circle"] else state
        logging = True if ps4.data["triangle"] else logging

        # print(ps4.data)
        if state == "manual":
            # parse the ps4 data
            # ensure that data is not very stale
            if(oldstate == "autonomous"):
                print("learning")
                curr_data["throttle"] = 90
                curr_data["steer"] = 90
                auto_model.train(saved_model_path="current_rl_model")
                auto_model.clear_buffers()

            elif ps4.data["timestamp"]-time.time() > config["tolerance_time"]:
                print("exceeded tolerance")
                curr_data["throttle"] = 90
                curr_data["steer"] = 90
            else:
                curr_data["throttle"] = ((int(ps4.data["ly"])-128)/128)*config["throttle_allowance"] + 90
                curr_data["steer"] = ((int(ps4.data["rx"])-128)/128)*config["steer_allowance"] + 90
            
                if logging:
                    log_buffer.append(curr_data)
                    if len(log_buffer)==config["samples_per_file"]:
                        p = Process(target=save_to_file, args=(config["log_path"], log_counter, deepcopy(log_buffer)))
                        p.start()
                        log_buffer = []
                        log_counter += 1

        elif state == "autonomous":
            if auto_model is not None:
                t_img = norm_split(Image.fromarray(img))
                throttle, steer = auto_model.run(t_img)
                curr_data["throttle"] = throttle
                curr_data["steer"] = steer
            else:
                curr_data["throttle"] = 90
                curr_data["steer"] = 90
            # curr_data["throttle"] = 90
            # curr_data["steer"] = 90

        else:
            raise ValueError(f"state {state} not implemented yet")

        print(curr_data["throttle"], curr_data["steer"])
        # print(state, curr_data)
        curr_data["timestamp"] = time.time()
        # send control to the motors
        motor.send_data(curr_data)

        if time.time() > (ts_start+config["refresh_time"]):
            print(f"main loop takes {time.time()-ts_start}")
        else:
            # busy wait while time requirement is met
            while time.time() < (ts_start+config["refresh_time"]):
                pass