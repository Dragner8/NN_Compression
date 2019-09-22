import numpy as np
import torch
import torch.nn as nn
from Pruner.utils import calculate_mask
from Models.c_resnet import resnet50,BottleNeck


def computeL1(m0, m1, top_idxs):

    amount = m1.weight.data.shape[0]

    weight = m0.weight.data

    weight_abs = weight.abs().detach().numpy().copy()
    sum_filter_val = np.sum(weight_abs, axis=(2, 3))
    l1_norm = np.sum(sum_filter_val, axis=(1))

    sorted_idx = np.argsort(l1_norm)
    top_idx = sorted_idx[::-1][:amount]
    top_idx = np.sort(top_idx)
    top_idxs.append(top_idx.tolist())


def prune_filters(model,cfg,cifar):
    newmodel = resnet50(cfg=cfg,num_class=cifar)
    print(newmodel)

    newmodel.to("cpu")

    top_idxs = []

    computeL1(model.conv1[0], newmodel.conv1[0], top_idxs)
    for [m0, m1] in zip(model.modules(), newmodel.modules()):

        if isinstance(m0, BottleNeck):
            for [m_0, m_1] in zip(m0.residual_function, m1.residual_function):
                if isinstance(m_0, nn.Conv2d):
                    computeL1(m_0, m_1, top_idxs)

    # PRUNE IN FIRST LAYER
    w = model.conv1[0].weight[top_idxs[0], :, :, :].clone()
    newmodel.conv1[0].weight.data = w.clone()
    newmodel.conv1[1].weight.data = model.conv1[1].weight[top_idxs[0]].clone()
    newmodel.conv1[1].bias.data = model.conv1[1].bias[top_idxs[0]].clone()
    newmodel.conv1[1].running_mean.data = model.conv1[1].running_mean[top_idxs[0]].clone()
    newmodel.conv1[1].running_var.data = model.conv1[1].running_var[top_idxs[0]].clone()

    def tmp(value):
        weight_abs = value.abs().detach().numpy().copy()
        sum_filter_val = np.sum(weight_abs, axis=(2, 3))
        l1_norm = np.sum(sum_filter_val, axis=(1))
        return l1_norm

    # Prune shortcuts
    # newmodel.conv2_x[0].shortcut[0].weight.data=model.conv2_x[0].shortcut[0].weight[top_idxs[3], top_idxs[0], :, :].clone()

    for i, m0, m1 in zip([[0, 3], [9, 12], [21, 24], [39, 42]],
                         [model.conv2_x[0], model.conv3_x[0], model.conv4_x[0], model.conv5_x[0]],
                         [newmodel.conv2_x[0], newmodel.conv3_x[0], newmodel.conv4_x[0], newmodel.conv5_x[0]]):


        sumL1 = tmp(m0.shortcut[0].weight.data) + tmp(m0.residual_function[6].weight.data)

        # weight_abs = sum.abs().detach().numpy().copy()
        # sum_filter_val = np.sum(weight_abs, axis=(2, 3))
        # l1_norm = np.sum(sum_filter_val, axis=(1))

        sorted_idx = np.argsort(sumL1)
        top_idx = sorted_idx[::-1][:len(top_idxs[i[1]])]
        top_idx = np.sort(top_idx)
        top_idxs[i[1]] = top_idx.tolist()

        w = m0.shortcut[0].weight.data[:, top_idxs[i[0]], :, :]
        w = w[top_idxs[i[1]], :, :, :]
        m1.shortcut[0].weight.data = w.clone()

        m1.shortcut[1].weight.data = m0.shortcut[1].weight.data[top_idxs[i[1]]].clone()
        m1.shortcut[1].bias.data = m0.shortcut[1].bias.data[top_idxs[i[1]]].clone()
        m1.shortcut[1].running_mean.data = m0.shortcut[1].running_mean.data[top_idxs[i[1]]].clone()
        m1.shortcut[1].running_var.data = m0.shortcut[1].running_var.data[top_idxs[i[1]]].clone()


    # newmodel.fc.weight.data = model.fc.weight.clone()
    # newmodel.fc.bias.data = model.fc.bias.clone()
    top_idxs = top_idxs[::-1]


    # Prune conv layers
    for [m0, m1] in zip(model.modules(), newmodel.modules()):

        if isinstance(m0, BottleNeck):
            for [m_0, m_1] in zip(m0.residual_function, m1.residual_function):
                if isinstance(m_0, nn.Conv2d):
                    w = m_0.weight[:, top_idxs[-1], :, :].clone()
                    w = w[top_idxs[-2], :, :, :].clone()
                    m_1.weight.data = w.clone()

                if isinstance(m_0, nn.BatchNorm2d):
                    m_1.weight.data = m_0.weight[top_idxs[-2]].clone()
                    m_1.bias.data = m_0.bias[top_idxs[-2]].clone()
                    m_1.running_mean.data = m_0.running_mean[top_idxs[-2]].clone()
                    m_1.running_var.data = m_0.running_var[top_idxs[-2]].clone()

                    top_idxs.pop()

    # PRUNE LINEAR

    for [m0, m1] in zip(model.modules(), newmodel.modules()):


        if isinstance(m0, nn.Linear):

            newmodel.fc.weight.data = model.fc.weight.data[:, top_idxs[-1]].clone()
            newmodel.fc.bias.data = model.fc.bias.data.clone()


    newmodel.to("cuda:0")
    return newmodel,cfg


def prune_weights(model, masks=None, percentage =None):
    if percentage:
        assert len(percentage) == 17 , "need 17 percentage value to prune"
    if masks:
        assert len(masks) == 17, "need 17 masks value to prune"
    if masks==None:
        masks = []

        counter = 0

        for layer in (model.modules()):

            if (isinstance(layer, BottleNeck)):
                ms=[]
                for l in layer.modules():
                    if isinstance(l,nn.Conv2d):

                        mask = calculate_mask(l, percentage[counter])
                        ms.append(mask)


                masks.append(ms)
                counter+=1

            if isinstance(layer, nn.Linear):
                mask = calculate_mask(layer, percentage[counter])
                masks.append(mask)
                counter += 1




    layers=[]
    for layer in (model.modules()):
        if (isinstance(layer, BottleNeck) or isinstance(layer, nn.Linear)):
            layers.append(layer)

    for layer, mask in zip(layers,masks):
        if isinstance(layer,BottleNeck):
            ms = mask

            count = 0
            for l in layer.modules():

                if isinstance(l,nn.Conv2d):


                    weight=l.weight.detach().cpu().numpy()

                    l.weight.data = torch.from_numpy(weight*ms[count].numpy()).clone()
                    count=count+1



        if isinstance(layer, nn.Linear):
            weight=(layer.weight.detach().cpu().numpy())
            mask=mask.numpy()
            layer.weight.data=torch.from_numpy(weight* mask)



    return model,masks
