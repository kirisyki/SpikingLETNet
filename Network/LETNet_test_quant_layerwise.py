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

def print_model(model):
    for name, child in model.named_children():
        if isinstance(child, QLayer):
            print(f"name: {child.name} weights: {child.layer.weight.data}")
        else:
            print_model(child)

def check_batchnorm_params(model):
    print("========== BatchNorm Layers ==========")
    bias_rec = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            print(f"\n--- Layer: {name} ({module.__class__.__name__}) ---")
            print(f"weight (gamma):       {module.weight.data.shape}")
            print(f"bias (beta):          {module.bias.data.shape}")
            print(f"running_mean:         {module.running_mean.shape}")
            print(f"running_var:          {module.running_var.shape}")
            print(f"num_batches_tracked:  {module.num_batches_tracked}")
            
            # print("weight values:", module.weight.data)
            # print("bias values:", module.bias.data)

            bias_rec.append(module.bias.data.detach().cpu().numpy().flatten())
    bias_rec = np.concatenate(bias_rec)
    plt.hist(bias_rec, bins=100, density=True)
    plt.xlabel("bias value")
    plt.ylabel("density")
    plt.title("bias value distribution")
    plt.savefig("./bias_distribution.png", dpi=600)

def plot_mid_outputs_hist(model, bins=50):
    for name, child in model.named_children():
        if isinstance(child, QLayer):
            all_mid_outputs = np.array(child.mid_output)
            save_path = os.path.join('/home/wyl/projects/LETNet/mid_outputs_plots', f'{name}.png')
            print(f"name: {name} mid_outputs_shape: {all_mid_outputs.shape}")
            plt.hist(all_mid_outputs, bins=bins, density=False)
            plt.xlabel('mid outputs value')
            plt.ylabel('Numbers')
            plt.title('mid outputs distribution')
            plt.grid()
            plt.savefig(save_path, dpi=600)
        else:
            plot_mid_outputs_hist(child, bins=50)


def plot_parameters_hist(model, bins=50):
    """
    可视化整个网络的参数分布直方图
    """
    all_params = []

    # 遍历所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_params.append(param.detach().cpu().numpy().flatten())

    # 拼接成一个大数组
    all_params = np.concatenate(all_params)

    # 计算稀疏度
    sparsity = np.sum(all_params == 0)/len(all_params)
    print(f"sparsity: {sparsity}")

    # 获取路径
    dir_name = os.path.dirname(args.checkpoint_q)
    save_path = os.path.join(dir_name, 'params_hist.png')

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(all_params, bins=bins, color='steelblue', alpha=0.75, density=True)
    plt.xlabel("Parameter value")
    plt.ylabel("Density")
    plt.title("Distribution of All Model Parameters")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    
def plot_bias(model):
    print("========== Checking Bias Distribution ==========")
    bias_rec = []
    for name, module in model.named_modules():
        if hasattr(module, 'bias') and module.bias is not None:
            bias_rec.append(module.bias.data.detach().cpu().numpy().flatten())
            print(name)
    bias_rec = np.concatenate(bias_rec)
    plt.hist(bias_rec, bins=100, density=True)
    plt.xlabel("bias value")
    plt.ylabel("density")
    plt.title("bias value distribution")
    plt.savefig("./bias_distribution.png", dpi=600)

def set_hardware_mode_layerwise(model, start_layer=0):
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, QLayer):
            if layer_idx >= start_layer:
                module.hardware_computing = True
            else:
                module.hardware_computing = False
            layer_idx += 1
    print(f"Set hardware computing mode from layer {start_layer} onwards. Total QLayers: {layer_idx}")

def set_hardware_mode_layerlist(model, layer_list):
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, QLayer):
            if layer_idx in layer_list:
                module.hardware_computing = True
            else:
                module.hardware_computing = False
            layer_idx += 1
    print(f"Set hardware computing mode for layers: {layer_list}. Total QLayers: {layer_idx}")

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
            if i >= 20:
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


    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

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

        # print("monitoring params distribution")
        # with open('params_distribution.txt', 'w') as f:
        #     for name, params in model.named_parameters():
        #         if params.requires_grad:
        #             mean = params.mean().item()
        #             std = params.std().item()
        #             f.write(f"layer_name: {name} mean: {mean} std:{std} \n")
        # f.close()
        # exit(-1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"total_params: {total_params}")
        print("start model fusing")
        # fuse_model(model, inplace=True)
        print("start model quantizing")
        model_q = quantize_model(model, k=4, quant=True)
        # print_model(model_q)
        if args.checkpoint_q:
            if args.checkpoint_q.endswith('.pth'):
                print("=====> loading quantized checkpoint '{}'".format(args.checkpoint_q))
                model_q.load_state_dict(torch.load(args.checkpoint_q)['model'])
            elif args.checkpoint_q.endswith('.pt'):
                print("=====> loading quantized checkpoint '{}'".format(args.checkpoint_q))
                model_q = torch.load(args.checkpoint_q, weights_only=False)
        with open('model_structure_fused.txt', 'w') as f:
            f.write(str(model_q))
        
        # plot_bias(model_q)
        # check_batchnorm_params(model_q)
        # plot_parameters_hist(model_q)
        f.close()
        model.eval()
        model_q.eval()

        # set computing mode of model_q
        set_hardware_mode_layerwise(model_q, start_layer=0)
        set_T_step(model_q, T=8)
        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        mIOU_val, per_class_iu = test(args, testLoader, model_q)
        print("mIOU_val:",mIOU_val)
        print("per_class_iu:",per_class_iu)

        # plot_mid_outputs_hist(model_q)

    # Get the best test result among the last 10 model records.
    else:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                dirname, basename = os.path.split(args.checkpoint)
                epoch = int(os.path.splitext(basename)[0].split('_')[1])
                mIOU_val = []
                per_class_iu = []
                for i in range(epoch - 9, epoch + 1):
                    basename = 'model_' + str(i) + '.pth'
                    resume = os.path.join(dirname, basename)
                    checkpoint = torch.load(resume)
                    model.load_state_dict(checkpoint['model'])
                    print("=====> beginning test the " + basename)
                    print("validation set length: ", len(testLoader))
                    mIOU_val_0, per_class_iu_0 = test(args, testLoader, model_q)
                    mIOU_val.append(mIOU_val_0)
                    per_class_iu.append(per_class_iu_0)

                index = list(range(epoch - 9, epoch + 1))[np.argmax(mIOU_val)] #选出最后10个模型mIoU最大的索引值
                print("The best mIoU among the last 10 models is", index)
                print(mIOU_val)
                per_class_iu = per_class_iu[np.argmax(mIOU_val)]
                mIOU_val = np.max(mIOU_val)
                print(mIOU_val)
                print(per_class_iu)

            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Mean IoU: %.4f" % mIOU_val)
        logger.write("\nPer class IoU: ")
        for i in range(len(per_class_iu)):
            logger.write("%.4f\t" % per_class_iu[i])
    logger.flush()
    logger.close()


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

