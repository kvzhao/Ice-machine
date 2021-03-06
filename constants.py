TASK_NAME = 'WGAN'
# Training Process

BATCH_SIZE = 128
NUM_BATCHES = 100000

D_ITERS = 5

EVAL_PER_ITERS   = 20
SAMPLE_PER_ITERS = 200
SAVE_CKPT_PER_ITERS = NUM_BATCHES / 20


# Data info

WIDTH = 64
HEIGHT = 64
CHANNEL = 1
Z_DIM = 128

# devices

DEVICE = '/gpu:0'

# Sovler parameters

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
BETA1_G = 0.5
BETA2_G = 0.9
BETA1_D = 0.5
BETA2_D = 0.9


# Names of ice dataset
DATASET_PATH = 'IcestateDataset'
DATASET_NAME = 'SQUAREICE_STATES_64x5000.h5'
IMAGE_NAME = 'ICESTATES'
GRAY_SCALE = True

##### TEST PARAMS: DELETE WHEN RUNNING #####

#EVAL_PER_ITERS   = 1
#SAMPLE_PER_ITERS = 1
#SAVE_CKPT_PER_ITERS = 2
#BATCH_SIZE = 16