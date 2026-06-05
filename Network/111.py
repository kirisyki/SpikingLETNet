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
from quantization.switch_computing_mode import set_computing_mode, set_T_step


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="SpikingLETNet_shallow_middle", help="model name: (default SpikingLETNet_shallow)")
    parser.add_argument('--dataset', default="udd", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=6, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="/home/wyl/projects/LETNet/QAT_checkpoint/udd/SpikingLETNet_shallow_middlebs32gpu1_trainval20260104-185754/model_best.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--checkpoint_q', type=str,default="/home/wyl/projects/LETNet/QAT_checkpoint/udd/SpikingLETNet_shallow_middlebs32gpu1_trainval20260104-185754/model_q_best_complete.pt",
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
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:
                break

            if args.dataset == 'voc' or args.dataset == 'udd':
                input, label = batch
            else:
                input, label, size, name = batch
            input_var, label = input.cuda(), label.cuda()
            start_time = time.time()
            output = 0
            input_var = input_var.unsqueeze(0)

            for j in range(args.T):
                output += model(input_var)
            output = output.mean(0)
            confm.update(label.flatten(), output.argmax(1).flatten())
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
            functional.reset_net(model)

            if args.save_results:
                print("Saving prediction results...")
                # 保存路径
                import matplotlib.pyplot as plt
                save_dir = os.path.join(args.save_seg_dir, 'vis')
                os.makedirs(save_dir, exist_ok=True)
                preds = output.argmax(1)[0]
                labels = label[0]
                pred_mask = preds.squeeze(0).cpu().numpy()
                gt_mask = labels.squeeze(0).cpu().numpy()
                print(f"pred_mask shape: {pred_mask.shape}, gt_mask shape: {gt_mask.shape}")
                # 保存灰度预测结果
                pred_gray_path = os.path.join(save_dir, f"pred_{i:03d}_gray.png")
                plt.imsave(pred_gray_path, pred_mask, cmap='gray')

                # 伪彩色可视化（使用随机颜色或自定义调色板）
                from matplotlib import colors
                cmap = colors.ListedColormap([
                    [0, 0, 0],         # background
                    [0, 0, 255],       # class 1
                    [0, 255, 0],       # class 2
                    [255, 0, 0],       # class 3
                    [255, 255, 0],     # class 4
                    [255, 0, 255],     # class 5
                    [0, 255, 255],     # class 6
                ])
                pred_color_path = os.path.join(save_dir, f"pred_{i:03d}_color.png")
                plt.imsave(pred_color_path, pred_mask, cmap=cmap)

                gt_color_path = os.path.join(save_dir, f"gt_{i:03d}.png")
                plt.imsave(gt_color_path, gt_mask, cmap=cmap)
            # output = output.cpu().data[0].numpy()
            # gt = np.asarray(label[0].numpy(), dtype=np.uint8)
            # output = output.transpose(1, 2, 0)
            # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            # data_list.append([gt.flatten(), output.flatten()])

            # save the predicted image
            if args.save:
                save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                            output_grey=False, output_color=True, gt_color=True)
    if args.dataset == 'voc':
        s_acc_global, s_acc, s_iou, s_f1 = confm.compute(with_background=True)
        s_acc, meanIoU, s_f1 = s_acc.mean().item(), s_iou.mean().item(), s_f1.mean().item()
        per_class_iou = s_iou.cpu().numpy().tolist()
        return meanIoU, per_class_iou
    elif args.dataset == 'udd':
        s_acc_global, s_acc, s_iou, s_f1 = confm.compute(with_background=True)
        s_acc = s_acc[:6]
        s_iou = s_iou[:6]
        s_f1 = s_f1[:6]
        s_acc, meanIoU, s_f1 = s_acc.mean().item(), s_iou.mean().item(), s_f1.mean().item()
        per_class_iou = s_iou.cpu().numpy().tolist()
        return meanIoU, per_class_iou


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        torch.cuda.set_device(int(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes, config=args.config)
    functional.set_step_mode(model, 'm')

    param_nums = sum(param.numel() for param in model.parameters())
    print(f"Model {args.model} has {param_nums} parameters")
    model_q = quantize_model(model, k=4, inplace=False, quant=True)
    with open(f"{args.model}_structure.txt", 'w') as f:
        f.write(str(model_q))
    


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'voc':
        args.classes = 21
    elif args.dataset == 'udd':
        args.classes = 6
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)


    test_model(args)

