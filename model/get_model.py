import os.path
import torch

from model.pytorch_i3d import InceptionI3d
from dotenv import load_dotenv


load_dotenv()

FILE_MODEL_EASY = os.getenv('FILE_MODEL_EASY')
FILE_MODEL_MEDIUM = os.getenv('FILE_MODEL_MEDIUM')
FILE_MODEL_HARD = os.getenv('FILE_MODEL_HARD')
NUM_CHANNELS = int(os.getenv('NUM_CHANNELS'))
NUM_CLASSES_EASY = int(os.getenv('NUM_CLASSES_EASY'))
NUM_CLASSES_MEDIUM = int(os.getenv('NUM_CLASSES_MEDIUM'))
NUM_CLASSES_HARD = int(os.getenv('NUM_CLASSES_HARD'))
DEVICE = os.getenv('DEVICE')

# Set up model  ----------------------------------------------------------------------------------------------------
i3d_easy = InceptionI3d(400, in_channels=NUM_CHANNELS)
i3d_easy.replace_logits(NUM_CLASSES_EASY)
i3d_easy.load_state_dict(torch.load(FILE_MODEL_EASY, map_location=torch.device(DEVICE)))
i3d_easy.eval()

i3d_medium = InceptionI3d(400, in_channels=NUM_CHANNELS)
i3d_medium.replace_logits(NUM_CLASSES_MEDIUM)
i3d_medium.load_state_dict(torch.load(FILE_MODEL_MEDIUM, map_location=torch.device(DEVICE)))
i3d_medium.eval()

i3d_hard = InceptionI3d(400, in_channels=NUM_CHANNELS)
i3d_hard.replace_logits(NUM_CLASSES_HARD)
i3d_hard.load_state_dict(torch.load(FILE_MODEL_HARD, map_location=torch.device(DEVICE)))
i3d_hard.eval()

models = {
    "EASY" : i3d_easy,
    "MEDIUM" : i3d_medium,
    "HARD" : i3d_hard
}


def get_model(difficulty):
    return models[difficulty]
