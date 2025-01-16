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

import networkx



def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

