# Imports --------------------------------------------------------------------------------------------------------------
import cv2
import os.path
import torch
import numpy as np

from model.difficulty import Difficulty
from model.get_model import get_model
from model.utils import videotransforms, utils
from dotenv import load_dotenv
from torchvision import transforms

# Consts ---------------------------------------------------------------------------------------------------------------
load_dotenv()

FILE_WEIGHTS = os.getenv('FILE_WEIGHTS')
VIDEO_DIRECTORY = os.getenv('VIDEO_DIRECTORY')
VIDEO_NAME = os.getenv('VIDEO_NAME')
VIDEO_EXTENSION = os.getenv('VIDEO_EXTENSION')


def predict(filename, difficulty = Difficulty.HARD):
    video = cv2.VideoCapture(filename)

    # Transform video --------------------------------------------------------------------------------------------------
    # Load video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Transform video
    start_f = 0
    rgb_frames = utils.load_rgb_frames_from_video(video, start_f, num_frames)
    padded_frames = utils.pad(rgb_frames, num_frames)
    crop_transformations = transforms.Compose([videotransforms.CenterCrop(224)])
    transformed_video_as_images = crop_transformations(padded_frames)

    # Create tensor
    transformed_video_as_tensor = utils.video_to_tensor(transformed_video_as_images)
    input_model = torch.unsqueeze(transformed_video_as_tensor, 0)

    # Predict
    i3d = get_model(difficulty)
    output = i3d(input_model)

    # Format result
    predictions = torch.max(output, dim=2)[0]
    predictions = predictions.cpu().detach().numpy()[0]
    out_index_labels = np.argsort(predictions)
    out_probs = np.sort(predictions)

    last_out_index_labels = np.take(out_index_labels, list(range(-10, 0)))
    last_out_probs = np.take(out_probs, list(range(-10, 0)))

    dictionary = utils.create_dictionary()
    last_out_labels = []
    for x in last_out_index_labels:
        last_out_labels.append(dictionary[x])

    return last_out_labels[::-1], last_out_probs[::-1]