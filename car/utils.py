import os
import pickle

def save_to_file(dname, counter, buff):
    if not os.path.exists(dname):
        os.makedirs(dname)
    filename = os.path.join(dname, "log_{}.pkl".format(counter))
    with open(filename, "wb") as f:
        pickle.dump(buff, f)