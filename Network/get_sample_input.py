import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from torchsummary import summary
from argparse import ArgumentParser
import sys
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict
from train_snn import ConfusionMatrix
from spikingjelly.activation_based import neuron, layer, functional
from quantization.model_fuse import fuse_model
from quantization.int4_selfbuild import quantize_model, check_frozen_scale_x, QLayer
from matplotlib import pyplot as plt


# all built on SpikingLETNet_shallow with udd
def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="SpikingLETNet_shallow_max", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="udd", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=6, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="/home/wyl/projects/LETNet/QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs32gpu1_trainval20251130-165315/model_best.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--checkpoint_q', type=str,default="/home/wyl/projects/LETNet/QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs32gpu1_trainval20251130-172826/model_q_best.pth",
                        help="use the file to load the quantized checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--config', type=str, default='/home/wyl/projects/LETNet/Network/configs/SpikingLETNet_shallow/1.3M.yaml')
    parser.add_argument('--best', action='store_true', default=False, help="Get the best result among last few checkpoints")
    parser.add_argument('--save', action='store_true', default=False, help="Save the predicted image")
    parser.add_argument('--save_results', action='store_true', default=False, help="Save the predicted image results")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="3", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--T', default=1, type=int, help="timesteps")
    args = parser.parse_args()

    return args

def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    model.to('cuda')
    total_batches = len(test_loader)

    confm = ConfusionMatrix(21)
    data_list = []
    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            input, batch = batch
            input = input.unsqueeze(0).cuda()
            print(f"input_shape:{input.shape}")
            init_conv_layer = model.init_conv[0]
            init_conv_output = init_conv_layer(input)
            return init_conv_output

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'voc':
        args.classes = 21
    elif args.dataset == 'udd':
        args.classes = 6

    model = build_model(args.model, num_classes=args.classes, config=args.config)
    functional.set_step_mode(model, 'm')

    model_q = quantize_model(model, k=4, quant=True)
    # print_model(model_q)
    if args.checkpoint_q:
        if args.checkpoint_q.endswith('.pth'):
            print("=====> loading quantized checkpoint '{}'".format(args.checkpoint_q))
            model_q.load_state_dict(torch.load(args.checkpoint_q)['model'])
        elif args.checkpoint_q.endswith('.pt'):
            print("=====> loading quantized checkpoint '{}'".format(args.checkpoint_q))
            model_q = torch.load(args.checkpoint_q, weights_only=False)

        # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if not args.best:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                print("=====> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model'])
                # model.load_state_dict(convert_state_dict(checkpoint['model']))
            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
            
    print("=====> beginning validation")
    init_conv_output = test(args, testLoader, model)
    print(f"init_conv_output_shape: {init_conv_output.shape}")
    accumulated_conv_output = init_conv_output.sum(0)
    sample_channel = accumulated_conv_output[0, 0:1, :, :].detach().cpu().numpy().flatten()
    # sample_channel = init_conv_output[0, 0, 0:3, :, :].detach().cpu().numpy().flatten()

    # 查看数值分布
    plt.hist(sample_channel, bins=50, density=True)
    plt.savefig('sample_channel_distribution_q.png', dpi=600)


