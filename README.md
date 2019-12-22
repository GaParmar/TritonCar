# TritonCar
Open source framework for autonomous RC car

## PS4 controls
 - **throttle:** left joystick up/down
 - **steer:** right joystick left/right
 - **autonomous mode:** circle
 - **manual mode:** cross
 - **start logging:** triangle
 - **stop logging:** square 

## Bluetooth framework
 - uses ds4drv
 - makes joystick device at */dev/input/js0*
 - starts a new screen on startup
 - can monitor ds4srv status by `screen -r ds4drv`
 - can monitor raw joystick data by `jstest /dev/input/js0`


## TODO
 - add raspi operating system image upload as dmg
 - inference mode switching between keras and pytorch
 - added setup steps