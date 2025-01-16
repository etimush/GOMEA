import PIL.Image
import numpy as np
import cv2
from numba import jit, njit
#from joblib import Parallel
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from scipy.stats import chi2_contingency
import time
import cProfile

from PIL import Image, ImageDraw
import csv
import  copy
import multiprocessing as mp
import random
import os; os.system('')
import torchvision.models as models
from torchvision import transforms
import torch

def calcEntropy(img):
    entropy = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
            en = en[0]
        entropy.append(en)


    sum_en = np.sum(entropy)
    return sum_en

model = models.convnext_small(pretrained=True)
model.eval()

def neural_net(img):
    input_image = Image.fromarray(img)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    return output[0][30]*-1


def cv2_norm(ground_t,painting):
    return cv2.norm( ground_t, painting, cv2.NORM_L2)