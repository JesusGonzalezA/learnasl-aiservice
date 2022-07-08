import math
import torch
import cv2
import os.path
import numpy as np
from dotenv import load_dotenv

load_dotenv()

TRANSPOSE_AXES = [3, 0, 1, 2]
VIDEO_EXTENSION = os.getenv('VIDEO_EXTENSION')
MIN_W = int(os.getenv('MIN_W'))
MIN_H = int(os.getenv('MIN_H'))
MIN = int(os.getenv('MIN'))
MAX_W = int(os.getenv('MAX_W'))
MAX_H = int(os.getenv('MAX_H'))
PROB = float(os.getenv('PROB'))
FILE_PREPROCESS = os.getenv('FILE_PREPROCESS')

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)

        Args:
             pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
             Tensor: Converted video.
    """
    video = pic.transpose(TRANSPOSE_AXES)
    return torch.from_numpy(video)

def load_rgb_frames_from_video(vidcap, start, num, resize=(256, 256)):
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()
        w, h, c = img.shape
        if w < MIN_W or h < MIN_H:
            d = MIN - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > MAX_W or h > MAX_H:
            img = cv2.resize(img, (math.ceil(w * (MAX_W / w)), math.ceil(h * (MAX_H / h))))

        img = (img / 255.) * 2 - 1
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

def pad(imgs, total_frames):
    if imgs.shape[0] < total_frames:
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > PROB:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    return padded_imgs

def create_dictionary():
    dictionary = []
    file = open(FILE_PREPROCESS, "r")
    for line in file:
        word = line.split()[1]
        dictionary.append(word)
    file.close()
    return dictionary