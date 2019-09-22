
import argparse
#from dataset import *
import os
#from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import Pruner.vgg_pruner as pruner
import Quantizer.vgg_quantizer as quantizer
#from conf import settings
from utils import cifar_test_dataloader

q=torch.ones(4).mul(2)
w=torch.ones(4).mul(3)
e=torch.ones(4)

dupa=torch.add(q,w)
dupa=torch.add(dupa,e)
print(dupa)

samples=torch.load("samples.tar")
print(samples['samples'][1])