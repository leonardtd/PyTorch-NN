import torch

DATA_DIR = "../dataset/"
BATCH_SIZE = 20
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
NUM_WORKERS = 8
EPOCHS = 30
NUM_EMBEDDINGS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

