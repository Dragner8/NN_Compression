import numpy as np
import torch
import torch.nn as nn
from Models.c_vgg import MLinear
import Models.vgg as vgg
from Pruner.utils import calculate_mask




def prune_filter(model,cfg,cifar):
    """"""

    #new_cfg =[16  , 64, 'M', 128, 128, 'M', 256, 256, 256,256, 'M', 128, 128, 64, 64, 'M', 128, 128, 128, 64, 'M']
    new_cfg =[16, 64, 'M', 128, 128, 'M', 256, 256, 128, 128, 'M', 128, 128, 256, 256, 'M', 128, 128, 128, 128, 'M']
    new_cfg=cfg

    #new_cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


    from Models.c_vgg import compress_vgg19_bn

    new_model = compress_vgg19_bn(num_class=cifar)
    conv_ids = []  

    for z, layer in (model.features._modules.items()):
        if (isinstance(layer, torch.nn.Conv2d)):
            conv_ids.append(int(z))

    conv_cfg = new_cfg.copy()
    for x in range(conv_cfg.count('M')):
        conv_cfg.remove('M')

    # calculate mask

    conv_masks = []
    for x in range(len(conv_ids)):
        mask = []
        weight = model.features[conv_ids[x]].weight

        weight_abs = model.features[conv_ids[x]].weight.abs().detach().numpy().copy()
        sum_filter_val = np.sum(weight_abs, axis=(2, 3))
        l1_norm = np.sum(sum_filter_val, axis=(1))

        sorted_idx = np.argsort(l1_norm)
        top_idx = sorted_idx[::-1][:conv_cfg[x]]

        mask = (top_idx.tolist())

        conv_masks.append(mask)
    #print(conv_masks)
    w = model.features[conv_ids[0]].weight[conv_masks[0], :, :, :].clone()
    b = model.features[conv_ids[0]].bias[conv_masks[0]].clone()

    new_model.features[conv_ids[0]].weight.data = w.clone()
    new_model.features[conv_ids[0]].bias.data = b.clone()

    new_model.features[1].weight.data = model.features[1].weight[conv_masks[0]].clone()
    new_model.features[1].bias.data = model.features[1].bias[conv_masks[0]].clone()
    new_model.features[1].running_mean.data = model.features[1].running_mean[conv_masks[0]].clone()
    new_model.features[1].running_var.data = model.features[1].running_var[conv_masks[0]].clone()

    for x in range(1, len(conv_ids)):
        w = model.features[conv_ids[x]].weight[:, conv_masks[x - 1], :, :].clone()
        w = w[conv_masks[x], :, :, :].clone()
        b = model.features[conv_ids[x]].bias[conv_masks[x]].clone()

        new_model.features[conv_ids[x]].weight.data = w.clone()
        new_model.features[conv_ids[x]].bias.data = b.clone()
        print(x)
        new_model.features[conv_ids[x]+1].weight.data = model.features[conv_ids[x]+1].weight[conv_masks[x]].clone()
        new_model.features[conv_ids[x]+1].bias.data = model.features[conv_ids[x]+1].bias[conv_masks[x]].clone()
        new_model.features[conv_ids[x]+1].running_mean.data = model.features[conv_ids[x]+1].running_mean[conv_masks[x]].clone()
        new_model.features[conv_ids[x]+1].running_var.data = model.features[conv_ids[x]+1].running_var[conv_masks[x]].clone()



    #copy FC values

    new_model.classifier[0].weight.data = model.classifier[0].weight.data[:, conv_masks[-1]].clone()
    new_model.classifier[0].bias.data = model.classifier[0].bias.data.clone()

    new_model.classifier[3].weight.data = model.classifier[3].weight.data.clone()
    new_model.classifier[3].bias.data = model.classifier[3].bias.data.clone()

    new_model.classifier[6].weight.data = model.classifier[6].weight.data.clone()
    new_model.classifier[6].bias.data = model.classifier[6].bias.data.clone()

    return new_model

def prune_weights(model, masks=None, percentage =None):
    if percentage:
        assert len(percentage) == 19 , "need 19 percentage value to prune"
    if masks:
        assert len(masks) == 19 , "need 19 masks value to prune"
    if masks==None:
        masks = []

        counter = 0
        for layer in (model.modules()):
            if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer,MLinear)):
                mask = calculate_mask(layer,percentage[counter])
                masks.append(mask)
                counter+=1
                pass

    layers=[]
    for layer in (model.modules()):
        if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, MLinear)):
            layers.append(layer)
    for layer, mask in zip(layers,masks):

        weight=(layer.weight.detach().cpu().numpy())
        mask=mask.numpy()

        layer.weight.data=torch.from_numpy(weight* mask)



    return model,masks



