# all configuration options to be defined here

# CAR
CAR_LOG_PATH = "/media/usb/LOG/lab335_0"
CAR_MODEL_PATH = "/media/usb/trained_model/cp_005_105.28.hdf5"
CAR_REFRESH_RATE = 0.04 # in Hz
CAR_CAMERA_ID = 0
CAR_TIMEOUT_TOLERANCE = 0.1 # in seconds
CAR_THROTTLE_ALLOWANCE = 25
CAR_STEER_ALLOWANCE = 35
CAR_SAMPLES_PER_FILE = 100 # num of samples for each log file
CAR_FIX_THROTTLE = 3 # -1 for disabling this


# COMMUNICATION
COMM_CONTROLLER_TYPE = "websocket_TCP" # websocket_TCP or bluetooth


# image settings
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
IMAGE_CH = 6

# TRAINING
TRAIN_DS_ROOT = "../OUTPUT/lab335/lab335"
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 4e-5

TRAIN_SU_outpath = f"output_models/"
TRAIN_SU_EXP_NAME = "exp0"