# Training Configuration

# Environment
NUM_ENVS = 4
MAX_EPISODE_STEPS = 1000
RANDOMIZE_RESET = True

# Training Duration
TOTAL_TIMESTEPS = 500000

# PPO Hyperparameters
LEARNING_RATE = 2e-4  # change back to 1e-4 when starting from scratch
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.02       # change back to 0.02 or 0.01 when starting from scratch
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Policy Network
# Note: Activation function is set in code (ReLU)
NET_ARCH_PI = [256, 256]
NET_ARCH_VF = [256, 256]
LOG_STD_INIT = -1.0

# Checkpoints & Evaluation
# Frequency in steps (will be divided by NUM_ENVS in code)
SAVE_FREQ_STEPS = 10000 
EVAL_FREQ_STEPS = 10000
N_EVAL_EPISODES = 10

# Paths
CHECKPOINT_DIR = "./duck_checkpoints/"
LOG_DIR = "./duck_tensorboard/"
BEST_MODEL_DIR = "./duck_best_model/"

# Walking
WALK_TARGET_VELOCITY = 0.5 # m/s
