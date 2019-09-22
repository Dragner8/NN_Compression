


import argparse

import numpy as np
import torch

import Pruner.vgg_pruner as pruner
import Quantizer.vgg_quantizer as quantizer
from utils import cifar_test_dataloader,load_compress_model,save_compress_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-cifar', type=int, default=10, choices=[10,100],
                      help='number of workers for dataloader')
    parser.add_argument('-prune', type=str, default=None, help='prune network')
    parser.add_argument('-quantize', type=str, default= "None", help='quantize network')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()



    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    cfg=None
    masks=None
    codebooks=None

    """ from Models.vgg  import  vgg19_bn
    cp=torch.load("smallModelsFC/cifar100/vgg19-163-best.pth")
    model=vgg19_bn()
    model.load_state_dict(cp)"""

    net, cfg, masks,codebooks = load_compress_model("Saved_models/prunedModels/cifar10/vgg.pth", cifar=args.cifar)
    model=net


    print(cfg)


    if args.prune:
        #new_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        #model=pruner.prune_filter(model,new_cfg,args.cifar)
        model, masks = pruner.prune_weights(model,percentage=[60,70,80,80,70,80,80,80,95,95,95,95,95,95,95,95,98,98,90])

    if args.quantize:
        model = quantizer.fixed_point_quant(model,bits=4)
        #quantize activations
        #model,codebooks = quantizer.kmeans_quant(model,k=8)
        # model = quantizer.linear_quant(model)

    cifar_test_loader = cifar_test_dataloader(
        args.cifar,
        CIFAR_MEAN,
        CIFAR_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )


    model.cuda()
    model.eval()


    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    import time
    start=time.time()
    for n_iter, (image, label) in enumerate(cifar_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar_test_loader)))
        image =image.cuda()
        label = label.cuda()
        output = model(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1
        correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 acc: ", correct_1 / len(cifar_test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(cifar_test_loader.dataset))
    print(time.time() - start)
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))


    ##SAVE MODEL

    #save_compress_model(model,'quant',cfg=cfg,mask=masks,codebook=codebooks)
    #print("saved model")