# TritonCar
Open source framework for autonomous RC car

## PS4 controls
 - **throttle:** left joystick up/down
 - **steer:** right joystick left/right
 - **autonomous mode:** circle
 - **manual mode:** cross
 - **start logging:** triangle
 - **stop logging:** square 

## Training steps
 - Copy the log pkl files from the RPI to local machine
 - `cd server && python3 prep_dataset.py --log_dir <path to folder>`
 - above command will extract images and labels from the pkl files

## Bluetooth framework
 - uses ds4drv
 - makes joystick device at */dev/input/js0*
 - starts a new screen on startup
 - can monitor ds4srv status by `screen -r ds4drv`
 - can monitor raw joystick data by `jstest /dev/input/js0`

## Web socket framework
 - socket communication is over port 8080
 - A packer is a byte string that starts with `<S>` and ends with `\n`
 - `utf-8` encoding is used
 - Open the port on SBC with the command `sudo ufw allow 8090 && sudo ufw enable && sudo reboot`
 - Launch the portal on the laptop with the command `python laptop/bt_server.py`
 - Open `http://127.0.0.1:5000/` in a browser window

## References
 - [Socket Communication](https://pythonprogramming.net/pickle-objects-sockets-tutorial-python-3/)
<!--  - 
 - packet size is a byte string that starts with <S> and ends with new line character
 - encoding utf8
 - open port on RPI with the commend 
 - launch server on the laptop `python3 laptop/bt_server.py`
 - browse to *http://127.0.0.1:5000/* in google chrome

## References
  - https://pythonprogramming.net/pickle-objects-sockets-tutorial-python-3/ -->
