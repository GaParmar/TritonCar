import os, sys, time
from PIL import Image
import numpy as np

src_folder = "/Users/gparmar/Desktop/github_gaparmar/TritonCar/server/OUTPUT/lab335"
out_folder = "left_32x16"
names = [im for im in os.listdir(src_folder) if im.endswith('.png')]

for name in names:
    im = Image.open(os.path.join(src_folder, name))
    w,h = im.size
    # crop to only the left image
    im = im.crop((0, 0, int(w/2), h))
    # crop away the top 1/3 of the height
    w,h = im.size
    im  = im.crop((0, int(h/3), w, h))
    # resize to 32x16 image
    im = im.resize((32,16)) 
    outname = os.path.join(out_folder, name)
    im.save(outname)