import os, sys, pdb
import time
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.22.207", 8080))

def parse_packet(pkt):
    data = {}
    pkt = pkt.decode("utf-8") 
    for token in pkt.split():
        if ":" in token:
            data[token.split(":")[0]] = token.split(":")[1]
    return data

while True:
    start = time.time()
    pkt = s.recv(128)
    print(parse_packet(pkt))
    while time.time()<(start+0.05):
        pass