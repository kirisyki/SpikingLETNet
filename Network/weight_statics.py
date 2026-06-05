import os
import time
import torch
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict
from train_snn import ConfusionMatrix
from spikingjelly.activation_based import neuron, layer, functional


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="SpikingLETNet_shallow_middle", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="udd", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=6, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="/home/met4physics/LETNet/Network/checkpoint/voc/SpikingLETNet_shallowbs16gpu1_trainval20250918-190845/model_best.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--save', action='store_true', default=False, help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--T', default=1, type=int, help="timesteps")
    parser.add_argument('--config', default="/home/wyl/projects/LETNet/Network/configs/SpikingLETNet_shallow/1.3M.yaml", type=str, help="model config")
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
    total_batches = len(test_loader)

    confm = ConfusionMatrix(21)
    data_list = []
    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            if args.dataset == 'voc':
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
            # output = output.cpu().data[0].numpy()
            # gt = np.asarray(label[0].numpy(), dtype=np.uint8)
            # output = output.transpose(1, 2, 0)
            # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            # data_list.append([gt.flatten(), output.flatten()])

            # save the predicted image
            if args.save:
                save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                            output_grey=False, output_color=True, gt_color=True)

    s_acc_global, s_acc, s_iou, s_f1 = confm.compute(with_background=True)
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
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes, config=args.config)
    functional.set_step_mode(model, 'm')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params/1e6:.2f} M")

    sample_input = torch.randn(1, 1, 3, 400, 400)
    # try:
    sample_output = model(sample_input)
    # except:
    #     print("the resized model can't work")
    #     exit(-1)
    
    print("the resized model can work.")




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
    
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)
    
    # the input size for network on UDD is 3*400*400
    test_model(args)
