# all configuration options to be defined here

# CAR
CAR_LOG_PATH = "/media/usb/LOG/lab335_0"
CAR_MODEL_PATH = "/media/usb/cp-012.hdf5"
CAR_REFRESH_RATE = 0.04 # in Hz
CAR_CAMERA_ID = 0
CAR_TIMEOUT_TOLERANCE = 0.1 # in seconds
CAR_THROTTLE_ALLOWANCE = 25
CAR_STEER_ALLOWANCE = 35
CAR_SAMPLES_PER_FILE = 100 # num of samples for each log file


# COMMUNICATION
COMM_CONTROLLER_TYPE = "websocket_TCP" # websocket_TCP or bluetooth
