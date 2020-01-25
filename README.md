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
 - socket communication over port 8080
 - packet size is a byte string that starts with <S> and ends with \n
 - encoding utf-8
 - open port on RPI with the commend `sudo ufw allow 8090 && sudo ufw enable && sudo reboot`
 - launch server on the laptop `python3 laptop/bt_server.py`
 - browse to *http://127.0.0.1:5000/* in google chrome


## Reinforcement Learning
 - train VAE: the state representation learning `cd server/VAE && python3 train.py <path_to_img_folder>`


## TODO
 - docker image for raspberry pi
 - docker image for training server - pytorch and keras versions
 - inference mode switching between keras and pytorch
 - added setup steps

 ### References
  - https://pythonprogramming.net/pickle-objects-sockets-tutorial-python-3/
  - https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4