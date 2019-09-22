import torch
import torch.nn as nn
from Quantizer.utils import fixed_point_tensor_quant,kmeans_tensor_quant
def fixed_point_quant(model,bits=8):

    for layer in (model.modules()):
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):

            quant = fixed_point_tensor_quant(layer.weight.cpu().detach().numpy(),bits)
            layer.weight.data = torch.from_numpy(quant).clone()

    return model


def kmeans_quant(model,k=8,codebooks=None):
    if codebooks:
        layers=[]
        for  layer in (model.modules()):
            if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
                layers.append(layer)
        for layer, codebook in zip(layers, codebooks):
                param = layer.weight.data
                codebook, param = kmeans_tensor_quant(param,k,codebook =codebook,update_labels=True)
                layer.weight.data = (param).clone()
                codebooks.append(codebook)
    else:
        codebooks=[]
        for layer in (model.modules()):
            if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
                param=layer.weight.data
                codebook,param = kmeans_tensor_quant(param,k)
                layer.weight.data = (param).clone()
                codebooks.append(codebook)

    return model,codebooks