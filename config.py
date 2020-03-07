# all configuration options to be defined here

# donkey_adapter or triton_car
MODE = "donkey_adapter"
CROP_TOP = 10
CROP_BOT = 50

# CAR
CAR_LOG_PATH = "/media/usb/LOG/lab335_0"
CAR_MODEL_PATH = "/media/usb/trained_model/cp_005_105.28.hdf5"
CAR_REFRESH_RATE = 0.1 # in Hz
CAR_CAMERA_ID = 0
CAR_TIMEOUT_TOLERANCE = 0.1 # in seconds
CAR_THROTTLE_ALLOWANCE = 25
CAR_STEER_ALLOWANCE = 35
CAR_SAMPLES_PER_FILE = 100 # num of samples for each log file
CAR_FIX_THROTTLE = -1 # -1 for disabling this


# COMMUNICATION
COMM_CONTROLLER_TYPE = "websocket_TCP" # websocket_TCP or bluetooth


# image settings
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
IMAGE_CH = 3

# TRAINING
TRAIN_DATASET_ROOT = "/content/tub_130_20-03-07"
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 128
TRAIN_LR = 1e-4
LAMBDA_STEER = 1.0

if MODE == "donkey_adapter":
    VAE_HEIGHT = IMAGE_WIDTH #180
    VAE_WIDTH = IMAGE_HEIGHT #320
else:
    VAE_HEIGHT = IMAGE_HEIGHT
    VAE_WIDTH = IMAGE_WIDTH
VAE_ZDIM = 16
VAE_LR = 1e-3
VAE_BATCH_SIZE = 32
VAE_EPOCHS = 100
VAE_LABEL = f"lab335_z{VAE_ZDIM}"
VAE_outpath = "output_models_vae"

TRAIN_SU_outpath = f"output_models/"
TRAIN_SU_EXP_NAME = "exp0"

## pilot training (after VAE training)
BEST_VAE_PATH = "../VAE/output_models/M_lab335_z32_1.sd"


PILOT_STEER_BINS = [ 90, # stay in the middle (90+0)
                    int(90-(CAR_STEER_ALLOWANCE*0.25)), int(90+(CAR_STEER_ALLOWANCE*0.25)),
                    int(90-(CAR_STEER_ALLOWANCE*0.50)), int(90+(CAR_STEER_ALLOWANCE*0.50)),
                    int(90-(CAR_STEER_ALLOWANCE*0.75)), int(90+(CAR_STEER_ALLOWANCE*0.75)),
                    int(90-(CAR_STEER_ALLOWANCE*1.00)), int(90+(CAR_STEER_ALLOWANCE*1.00))]

PILOT_THROTTLE_BINS = [ 90, # stay in the middle (90+0)
                    int(90-(CAR_THROTTLE_ALLOWANCE*0.25)), int(90+(CAR_THROTTLE_ALLOWANCE*0.25)),
                    int(90-(CAR_THROTTLE_ALLOWANCE*0.50)), int(90+(CAR_THROTTLE_ALLOWANCE*0.50)),
                    int(90-(CAR_THROTTLE_ALLOWANCE*0.75)), int(90+(CAR_THROTTLE_ALLOWANCE*0.75)),
                    int(90-(CAR_THROTTLE_ALLOWANCE*1.00)), int(90+(CAR_THROTTLE_ALLOWANCE*1.00))]