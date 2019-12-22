import os
import sys
import numpy as np
import pickle 
import argparse
import pdb

from PIL import Image

parser = argparse.ArgumentParser(description="unpack log pickle files")
parser.add_argument("--log_dir", type=str, 
                    required=True,
                    help="path to log pkl files")
parser.add_argument("--output_dir", type=str,
                    default="OUTPUT",
                    help="the output directory with unpacked samples")

args = parser.parse_args()

# output dir
args.output_dir = os.path.join(args.output_dir,
                    os.path.basename(os.path.abspath(args.log_dir)))

log_files = []
# get all pickle files
for file in os.listdir(args.log_dir):
    if ".pkl" in file:
        log_files.append(os.path.join(args.log_dir, file))

# make the output dir if it does not exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

g_ctr = 1
disconnected_frames = 0

for log_file in log_files:
    # unpickle the list of samples
    with open(log_file, "rb") as file1:
        print(log_file)
        samples = pickle.load(file1)
        for sample in samples:
            ts = sample["timestamp"]
            img = Image.fromarray(sample["image"])
            throttle = sample["throttle"]
            steer = sample["steer"]
            img = img.rotate(180)
            if(throttle != 90 or steer != 90):
                impath = os.path.join(args.output_dir,
                f"{g_ctr}_{ts:.3f}_{throttle}_{steer}.png")
                img.save(impath)
                g_ctr += 1
            else:
                disconnected_frames += 1

print("finished with {} frames disconnected and {} frames processed".format(disconnected_frames, g_ctr))