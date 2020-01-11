import os, sys, time, pdb
from flask import Flask, render_template, json, request
import socket
import pickle

app = Flask(__name__,static_url_path='')
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 8080

@app.route('/<path:path>/css')
def send_js(path):
    return send_from_directory('css', path)

@app.route('/<path:path>/js')
def send_css(path):
    return send_from_directory('js', path)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/data', methods = ['POST'])
def data():
    data = request.get_json()
    print(data)
    msg = "<S> "
    for key in data:
        print(key, data[key])
        msg+=f"{key}:{int(data[key]):03d} "
    msg += "\n"
    print(msg)
    # print(len(pkl_data))
    cs.sendall(msg.encode('utf-8'))
    return(json.dumps({}))

if __name__ == "__main__":
    global cs
    print(f"INITIALIZING TCP socket connection at port {PORT}\n\n")
    s.bind(("", PORT))
    s.listen()
    cs, address = s.accept()
    print(f"Connection established at {address}\n\n")
    app.run()