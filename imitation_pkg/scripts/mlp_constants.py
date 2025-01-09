import torch

# MLP constants
INPUT_LAYER_SIZE = 216
HIDDEN_LAYER_SIZE = 512
OUTPUT_LAYER_SIZE = 2
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 55

# Dataset to collect and train
DATASET_TO_COLLECT = "redbull_ring_gap_follow"
TRAIN_DATASET = "redbull_ring_gap_follow"

# Model to train and run
MODEL_TO_TRAIN = f"{TRAIN_DATASET}_{INPUT_LAYER_SIZE}_{HIDDEN_LAYER_SIZE}"
MODEL_TO_RUN = "IL_Model"