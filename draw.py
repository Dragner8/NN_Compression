import torch
import numpy as np
import matplotlib.pyplot as plt
def plot_w(model):
    weights = ((model.classifier[0].weight))
    weights = weights.cpu().detach().numpy()

    # a = numpy.random.uniform(-1, 1, size=10000)
    weights = np.around(weights,2)
    weights.sort()
    unique, counts = np.unique(weights, return_counts=True)
    for i in range(len(unique)):
        if (unique[i] == 0):
            counts[i] = 0

    plt.ylabel('Number of params')
    plt.xlabel('Values')
    plt.title('VGG cifar100 fc1 weight distribution')

def plot_w2(model):
    weights =(model.features[7].weight) #((model.conv4_x[0].residual_function[0].weight.data))
    weights = weights.cpu().detach().numpy()

    weights.sort()
    unique, counts = np.unique(weights, return_counts=True)
    for i in range(len(unique)):
        if (unique[i] == 0):
            counts[i] = 0
    plt.plot(unique, counts, label='line 1', linewidth=1)
    plt.show()

    y_pos = np.arange(len(unique))
    plt.bar(y_pos, counts, align='center', color="blue")
    plt.xticks(y_pos, unique)
    plt.ylabel('Number of params')
    plt.xlabel('Values')
    plt.title('6 bit quantized and pruned VGG cifar100 fc2 weight distribution')
    plt.show()

def plot_L(model):
    mask = []
    # weight = model.features[0].weight
    # start_size=weight.data.shape[0]
    weight_abs = model.classifier[3].weight.abs().detach().numpy().copy()
    sum_filter_val = np.sum(weight_abs, axis=(2, 3))
    print(sum_filter_val)
    l1_norm = np.sum(sum_filter_val, axis=(1))
    sorted = np.sort(l1_norm)
    sorted_idx = np.argsort(l1_norm)
    top_idx = sorted_idx[::-1]



    # plt.plot(unique, counts, label='line 1', linewidth=1)
    # plt.show()

    y_pos = np.arange(len(sorted))
    plt.bar(y_pos, sorted, align='center', color="red")
    plt.xticks(y_pos, top_idx)
    plt.ylabel('L1_norm')
    plt.xlabel('Filter number')
    plt.title('Filter pruning vgg19 conv1 cifar10 ')
    plt.show()


from utils import cifar_test_dataloader, load_compress_model, save_compress_model

#net, cfg, masks = load_compress_model("../checkpoint/pVG10/2019-09-02T12:44:55.800618/pVG10-30-best.pth", cifar=10)
# net, cfg, masks = load_compress_model("../smallModels/cifar10/final-4bit-small.tar", cifar=10)  #prunedModels/cifar10/vgg.pth
# print(net)
# plot_w(net)
# torch.set_printoptions(threshold=5000)
# print(net.classifier[3].weight)
#
#

#net, cfg, masks,codebooks = load_compress_model("checkpoint/qq/2019-09-16T21:46:23.930447/qq-2-best.pth", cifar=10,net="vgg")  #prunedModels/cifar10/vgg.pth
#print(net.conv4_x[0].residual_function[0].weight.data)
#plot_w2(net)
#
#
#
# net, cfg, masks = load_compress_model("../checkpoint/pVG10/2019-09-02T13:39:28.102492/pVG10-29-best.pth", cifar=10)
#
# plot_w(net)

import torch.nn as nn
net, cfg, masks,codebooks = load_compress_model("Saved_models/prunedModels/cifar10/resnet.pth", cifar=10,net="resnet")
params=0
# counter=0
# wagi=net2.classifier[3].weight
# for w in wagi.reshape(-1):
#     if w == 0 or w== -0 :
#         counter+=1
# print(counter)
# print(len(wagi.reshape(-1)))
# print(len(wagi.reshape(-1))-counter)
# print("DUPA")
# params=0
# left=0
# for layer in net.modules():
#     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#         counter = 0
#         wagi = layer.weight
#         for w in wagi.reshape(-1):
#             if w == 0 or w == -0:
#                 counter += 1
#
#         left+=counter
#         #print(len(wagi.reshape(-1))-counter)
#         counter=0
#         params+=len(wagi.reshape(-1))
#         print(len(wagi.reshape(-1)))
#
#
# print(params)
# print(left)


from Models.c_resnet import resnet50

model = resnet50(cfg=[[64], 3 * [64], 4 * [128], 6 * [256], 3 * [512]], num_class=100)
model.load_state_dict(torch.load("Saved_models/cifar100/resnet50-198-best.pth"))
net=model
#net, cfg, masks,codebooks = load_compress_model("Saved_models/orginalModels/cifar10/vgg.pth", cifar=10,net="vgg")
from Models.vgg import vgg19_bn
model = vgg19_bn()
model.load_state_dict(torch.load("smallModelsFC/cifar100/vgg19-163-best.pth"))
for layer in model.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):

        wagi = layer.weight


        params+=len(wagi.reshape(-1))
        print(len(wagi.reshape(-1)))


print(params)