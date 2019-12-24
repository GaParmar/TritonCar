import os
import pickle

def parse_packet(pkt, data):
    pkt = pkt.decode("utf-8") 
    for token in pkt.split():
        if ":" in token:
            key = token.split(":")[0]
            value = token.split(":")[1]
            if key in ["cross", "square", "triangle", "circle"]:
                value = value == "001"
            else:
                value = int(value)
            data[key] = value

    return data

def save_to_file(dname, counter, buff):
    if not os.path.exists(dname):
        os.makedirs(dname)
    filename = os.path.join(dname, "log_{}.pkl".format(counter))
    with open(filename, "wb") as f:
        pickle.dump(buff, f)