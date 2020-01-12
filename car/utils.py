import os
import pickle

def parse_packet(pkt):
    data = {}
    pkt = pkt.decode("utf-8") 
    for token in pkt.split():
        if ":" in token:
            data[token.split(":")[0]] = token.split(":")[1]
    return data

def save_to_file(dname, counter, buff):
    if not os.path.exists(dname):
        os.makedirs(dname)
    filename = os.path.join(dname, "log_{}.pkl".format(counter))
    with open(filename, "wb") as f:
        pickle.dump(buff, f)