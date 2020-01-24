import os, sys, time, pdb
import subprocess
from flask import Flask, render_template, json, request
import socket
import pickle

app = Flask(__name__,static_url_path='')
cs, s = None, None

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
    # cs is None if socket connection is not established
    if cs is None:
        return json.dumps({})
    data = request.get_json()
    print(data)
    msg = "<S> "
    for key in data:
        msg+=f"{key}:{int(data[key]):03d} "
    msg += "\n"
    try:
        cs.sendall(msg.encode('utf-8'))
        return("success")
    except:
        return ("failure")

@app.route("/get_host_ip", methods=["POST"])
def get_ip():
    host_ip = socket.gethostbyname(socket.gethostname())
    return(str(host_ip))

@app.route("/init_connection", methods=["POST"])
def init_socket():
    global s, cs, PORT
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data = request.get_json()

    print(f"INITIALIZING TCP socket connection at port {data['port']}\n")
    s.bind(("", int(data["port"])))
    s.listen()
    cs, address = s.accept()
    host_ip = socket.gethostbyname(socket.gethostname())
    return(json.dumps({"status": "success", "ip":host_ip}))

if __name__ == "__main__":
    app.run()