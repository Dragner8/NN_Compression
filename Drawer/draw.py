import torch
import numpy as np
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

    import matplotlib.pyplot as plt

    plt.ylabel('first FC quantized tp 5 bits')
    plt.xlabel('sorted active users')

    plt.plot(unique, counts, label='line 1', linewidth=1)
    plt.show()

    y_pos = np.arange(len(unique))
    plt.bar(y_pos, counts, align='center', color="blue")
    plt.xticks(y_pos, unique)
    plt.ylabel('articels')
    plt.xlabel('years')
    plt.title('Amount of articles')
    plt.show() \

from utils import cifar_test_dataloader, load_compress_model, save_compress_model

#net, cfg, masks = load_compress_model("../checkpoint/pVG10/2019-09-02T12:44:55.800618/pVG10-30-best.pth", cifar=10)
net, cfg, masks = load_compress_model("../smallModels/cifar10/final-4bit-small.tar", cifar=10)  #prunedModels/cifar10/vgg.pth
print(net)
plot_w(net)
# torch.set_printoptions(threshold=5000)
# print(net.classifier[3].weight)
#
#
print(net)
net2, cfg , masks= load_compress_model("../smallModels/cifar10/vgg.pth", cifar=10)
plot_w(net2)
#
#
#
# net, cfg, masks = load_compress_model("../checkpoint/pVG10/2019-09-02T13:39:28.102492/pVG10-29-best.pth", cifar=10)
#
# plot_w(net)


"""counter=0
wagi=net.classifier[3].weight
for w in wagi.reshape(-1):
    if w == 0 or w== -0 :
        counter+=1
print(counter)
print(len(wagi.reshape(-1)))
"""
