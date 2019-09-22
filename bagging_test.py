


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
from utils import cifar_test_dataloader,load_compress_model,save_compress_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-cifar', type=int, default=10, choices=[10,100],
                      help='number of workers for dataloader')
    parser.add_argument('-prune', type=str, default=None, help='prune network')
    parser.add_argument('-quantize', type=str, default=None, help='quantize network')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()



    CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    #model1, cfg1 ,masks1, codebooks1= load_compress_model("checkpoint/bag_vgg/2019-09-17T19:00:28.961866/b_vgg1-7-best.pth", cifar=args.cifar)
    #model2, cfg2 ,masks2, codebooks1= load_compress_model("checkpoint/bag_vgg/2019-09-17T19:00:28.961866/b_vgg2-7-best.pth", cifar=args.cifar)
    #model3, cfg3 = load_compress_model("checkpoint/bag_compress_vgg/2019-08-07T03:42:05.277633/q_compress_vgg3-15-best.pth", cifar=args.cifar)
    """    model1, cfg1, masks1, codebooks1 = load_compress_model("checkpoint/bag_vgg/2019-09-17T23:04:11.720876/b_vgg1-7-best.pth", cifar=args.cifar)
    model2, cfg2, masks2, codebooks2 = load_compress_model("checkpoint/bag_vgg/2019-09-17T23:04:11.720876/b_vgg2-7-best.pth", cifar=args.cifar)
    model3, cfg3, masks3, codebooks3 = load_compress_model("checkpoint/bag_vgg/2019-09-17T23:04:11.720876/b_vgg3-7-best.pth", cifar=args.cifar)
    model4, cfg4, masks4, codebooks4 = load_compress_model("checkpoint/bag_vgg/2019-09-17T23:04:11.720876/b_vgg4-7-best.pth", cifar=args.cifar)"""
    model1, cfg1, masks1, codebooks1 = load_compress_model("checkpoint/bag_vgg/2019-09-18T11:52:51.821220/b_vgg1-13-best.pth", cifar=args.cifar)
    model2, cfg2, masks2, codebooks2 = load_compress_model("checkpoint/bag_vgg/2019-09-18T11:52:51.821220/b_vgg2-13-best.pth", cifar=args.cifar)
    model3, cfg3, masks3, codebooks3 = load_compress_model("checkpoint/bag_vgg/2019-09-18T11:52:51.821220/b_vgg3-13-best.pth", cifar=args.cifar)
    model4, cfg4, masks4, codebooks4 = load_compress_model("checkpoint/bag_vgg/2019-09-18T11:52:51.821220/b_vgg4-12-best.pth", cifar=args.cifar)

    # model1, cfg1, masks1, codebooks1 = load_compress_model( "checkpoint/bag_vgg/2019-09-18T18:28:28.087751/b_vgg1-7-best.pth", cifar=args.cifar)
    # model2, cfg2, masks2, codebooks2 = load_compress_model("checkpoint/bag_vgg/2019-09-18T18:28:28.087751/b_vgg2-6-best.pth", cifar=args.cifar)
    # model3, cfg3, masks3, codebooks3 = load_compress_model("checkpoint/bag_vgg/2019-09-18T18:28:28.087751/b_vgg3-6-best.pth", cifar=args.cifar)
    # model4, cfg4, masks4, codebooks4 = load_compress_model( "checkpoint/bag_vgg/2019-09-18T18:28:28.087751/b_vgg4-6-best.pth", cifar=args.cifar)
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()
    # new_cfg = [16, 64, 'M', 128, 128, 'M', 256, 256, 128, 128, 'M', 128, 128, 256, 256, 'M', 128, 128, 128, 128, 'M']
    if args.prune:

        pass
        #new_model=vgg19_bn_pr(new_cfg)
        model1=pruner.prune(model1,cfg1)

    if args.quantize:
        model1 = quantizer.linear_quant(model1)
        model2 = quantizer.linear_quant(model2)
       # model3 = quantizer.linear_quant(model3)


        # model, index_model ,shared_w,shared_f= quantizer.weight_share_quant(model)
        #
        # import time
        # start=time.time()
        # model = quantizer.update_weight_from_codebook(model,index_model,shared_w,shared_f)
        # print(time.time()-start)
    cifar_test_loader = cifar_test_dataloader(
        args.cifar,
        CIFAR_MEAN,
        CIFAR_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    model1.cuda()
    model1.eval()
    model2.cuda()
    model2.eval()


    #model3.cuda()
    #model3.eval()


    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar_test_loader)))
        image =image.cuda()
        label = label.cuda()
        output1 = model1(image)
        output2 = model2(image)
        output3 = model3(image)
        output4 = model4(image)
        output=torch.add(output2,output1)
        output=torch.add(output,output3)
        output = torch.add(output, output4)
        #output = model1(image)
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


    ##SAVE MODEL
    #torch.save({'cfg': new_cfg, 'state_dict': model.state_dict()}, os.path.join(".", 'prune.pth.tar'))

    #save_compress_model(model,'quant_step4.pth',cfg=cfg,mask="")