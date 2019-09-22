import torch
from Quantizer.utils import linear_tensor_quant, fixed_point_tensor_quant, kmeans_tensor_quant
import torch.nn as nn
import numpy as np
import time
from Models.c_vgg import MLinear

def linear_quant(model):

    # quant Conv
    conv_ids = []
    for z, layer in (model.features._modules.items()):
        if (isinstance(layer, nn.Conv2d)) :
            conv_ids.append(int(z))

    for x in range(len(conv_ids)):
        quant = linear_tensor_quant(model.features[conv_ids[x]].weight.cpu().detach().numpy(), 0.2,-0.2,4)
        model.features[conv_ids[x]].weight.data = torch.from_numpy(quant).clone()

    # quant Linear
    quant = linear_tensor_quant(model.classifier[0].weight.detach().numpy(), 0.2,-0.2,4)
    model.classifier[0].weight.data = torch.from_numpy(quant).clone()

    quant = linear_tensor_quant(model.classifier[3].weight.detach().numpy(), 0.2,-0.2,4)
    model.classifier[3].weight.data = torch.from_numpy(quant).clone()

    quant = linear_tensor_quant(model.classifier[6].weight.detach().numpy(), 0.2,-0.2,4)
    model.classifier[6].weight.data = torch.from_numpy(quant).clone()
    return model

def fixed_point_quant(model,bits=3):

    # quant Conv
    conv_ids = []
    for id, layer in (model.features._modules.items()):
        if (isinstance(layer, nn.Conv2d)):
            conv_ids.append(int(id))

    for x in range(len(conv_ids)):
        quant = fixed_point_tensor_quant(model.features[conv_ids[x]].weight.cpu().detach().numpy(),bits)
        model.features[conv_ids[x]].weight.data = torch.from_numpy(quant).clone()

    # quant Linear
    quant = fixed_point_tensor_quant(model.classifier[0].weight.detach().numpy(), bits)
    model.classifier[0].weight.data = torch.from_numpy(quant).clone()

    quant = fixed_point_tensor_quant(model.classifier[3].weight.detach().numpy(), bits)
    model.classifier[3].weight.data = torch.from_numpy(quant).clone()

    quant = fixed_point_tensor_quant(model.classifier[6].weight.detach().numpy(), bits)
    model.classifier[6].weight.data = torch.from_numpy(quant).clone()
    return model

def kmeans_quant(model,k=4,codebooks=None):
    if codebooks:
        layers=[]
        for  layer in (model.modules()):
            if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
                layers.append(layer)
        for layer, codebook in zip(layers, codebooks):
                param = layer.weight.data
                codebook, param = kmeans_tensor_quant(param, k ,codebook =codebook,update_labels=True)
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
def weight_share_quant(model):

    # create model to save indices
    from Models.vgg import vgg19_bn, vgg19_bn_pr
    index_model=vgg19_bn()

    # quantiuze fc layers
    fc_ids=[]
    for i, layer in (model.classifier._modules.items()):
        if (isinstance(layer, nn.Linear) or isinstance(layer, MLinear) ):
            fc_ids.append(int(i))


    fc_unique_weight = np.array([])
    quants=[]
    for idx in fc_ids:
        quant = linear_tensor_quant(model.classifier[idx].weight.detach().numpy(), 0.24, -0.24, 8)
        model.classifier[idx].weight.data = torch.from_numpy(quant).clone()
        quants.append(quant)
        unique, counts = np.unique(quant, return_counts=True)
        fc_unique_weight = np.append(fc_unique_weight, unique.ravel())


    unique, counts = np.unique(fc_unique_weight, return_counts=True)
    shared_w = np.sort(unique)

    shared_w = shared_w.tolist()

    # save indexes in index_model
    pointer_weight = []
    tmp = []
    #print(len( quants[1][0]))

    start = time.time()
    for i in range(len(fc_ids)):
        quant=quants[i]
        id=fc_ids[i]
        for q in quant:
            for val in q:
                tmp.append(shared_w.index(val))
            pointer_weight.append(tmp)
            tmp = []
        index_model.classifier[id].weight.data = torch.from_numpy(np.asarray(pointer_weight)).type(torch.FloatTensor).clone()
        pointer_weight = []

    print(time.time() - start)

    # quant convolutions
    conv_ids = []
    filters = []
    conv_unique_weight = np.array([])
    for i, layer in (model.features._modules.items()):
        if (isinstance(layer, nn.Conv2d)):
            conv_ids.append(int(i))

    for x in range(len(conv_ids)):
        quant = linear_tensor_quant(model.features[conv_ids[x]].weight.cpu().detach().numpy(), 0.24, -0.24, 8)
        filters.append(quant)
        unique, counts = np.unique(quant, return_counts=True)
        conv_unique_weight = np.append(conv_unique_weight, unique.ravel())

        model.features[conv_ids[x]].weight.data = torch.from_numpy(quant).clone()
    unique, counts = np.unique(conv_unique_weight, return_counts=True)
    shared_f = np.sort(unique)  # .tolist()

    # save indexes in index_model

    start = time.time()
    tmp = filters.copy()  # np.ones((len(conv_ids),len(filters),len(filters[0]),len(filters[0][0]),len(filters[0][0][0])))
    for x in range(len(conv_ids)):

        shapes = [len(filters[x]), len(filters[x][0]), len(filters[x][0][0]), len(filters[x][0][0][0])]
        import itertools
        loopover = [range(s) for s in shapes]


        prod = itertools.product(*loopover)

        for idx in prod:
            i0, i1, i2, i3 = idx
            result = np.where(shared_f == filters[x][i0][i1][i2][i3])
            tmp[x][i0][i1][i2][i3] = result[0][0]

            # tmp[x][i0][i1][i2][i3]=shared_f.index(filters[x][i0][i1][i2][i3])

    i = 0
    for conv in (conv_ids):
        index_model.features[conv].weight.data = torch.from_numpy(tmp[i]).type(torch.FloatTensor).clone()
        i += 1
    print(time.time() - start)
    print(shared_w)
    print(shared_f)

    return model,index_model, shared_w,shared_f

def update_weight_from_codebook(model,index_model,shared_w,shared_f):

    # change fc weight to quantize values
    for i, m in zip(index_model.modules(), model.modules()):

        if isinstance(m, nn.Linear) or isinstance(m, nn.Linear):  # fc
            orginal = []
            tmp = []
            index = i.weight.data.type(torch.IntTensor)
            index = index.tolist()
            for vector in index:
                for value in vector:
                    # print(int(value.data.item()))
                    tmp.append(shared_w[(value)])

                orginal.append(tmp)
                tmp = []
            m.weight.data = torch.from_numpy(np.asarray(orginal)).type(torch.FloatTensor).clone()


    # change conv weight to quantize values
    for i, m in zip(index_model.modules(), model.modules()):
        if isinstance(m, nn.Conv2d) :
            index = i.weight.type(torch.IntTensor)
            index = index.tolist()
            weight=m.weight.data
            shapes = [len(weight), len(weight[0]), len(weight[0][0]), len(weight[0][0][0])]
            import itertools
            loopover = [range(s) for s in shapes]
            prod = itertools.product(*loopover)
            for idx in prod:
                i0, i1, i2, i3 = idx
                weight[i0][i1][i2][i3] = shared_f[index[i0][i1][i2][i3]]
            m.weight.data = torch.from_numpy(np.asarray(weight)).type(torch.FloatTensor).clone()




    return model








