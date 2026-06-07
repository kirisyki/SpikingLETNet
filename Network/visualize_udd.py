import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.convert_state import convert_state_dict
import torch.backends.cudnn as cudnn
from spikingjelly.activation_based import neuron, layer, functional
from quantization.switch_computing_mode import set_computing_mode, set_T_step
from quantization.int4_selfbuild import quantize_model, check_frozen_scale_x, QLayer
from model.module.neuron import QIFNode

# UDD 数据集的颜色映射 (根据 udd_preprocess.py 中的定义)
UDD_COLORS = {
    0: (0, 0, 0),        # Other - 黑色
    1: (102, 102, 156),  # Facade - 蓝紫色
    2: (128, 64, 128),   # Road - 紫色
    3: (107, 142, 35),   # Vegetation - 绿色
    4: (0, 0, 142),      # Vehicle - 深蓝色
    5: (70, 70, 70)      # Roof - 深灰色
}

UDD_CLASS_NAMES = {
    0: "Other",
    1: "Facade", 
    2: "Road",
    3: "Vegetation",
    4: "Vehicle",
    5: "Roof"
}

NUM_CLASSES = len(UDD_CLASS_NAMES)

def udd_colorize_mask(mask):
    """
    将UDD分割掩码转换为彩色图像
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in UDD_COLORS.items():
        color_mask[mask == class_id] = color
    
    return color_mask

def visualize_prediction(image, prediction, ground_truth=None, save_path=None, show=True):
    """
    可视化预测结果
    
    Args:
        image: 原始输入图像 (H, W, 3)
        prediction: 预测的分割掩码 (H, W)
        ground_truth: 真实的分割掩码 (H, W)，可选
        save_path: 保存路径，可选
        show: 是否显示图像
    """
    # 将预测结果转换为彩色
    pred_color = udd_colorize_mask(prediction)
    
    # 创建可视化图像
    if ground_truth is not None:
        gt_color = udd_colorize_mask(ground_truth)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 真实标签
        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # 预测结果
        axes[2].imshow(pred_color)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
    else:
        # 只有预测结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 预测结果
        axes[1].imshow(pred_color)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_legend():
    """创建UDD类别的图例"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    
    # 创建图例
    legend_elements = []
    for class_id in range(len(UDD_CLASS_NAMES)):
        color = np.array(UDD_COLORS[class_id]) / 255.0
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                           label=UDD_CLASS_NAMES[class_id]))
    
    ax.legend(handles=legend_elements, loc='center', ncol=3, 
              frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('udd_legend.png', dpi=300, bbox_inches='tight')
    print("Saved legend to udd_legend.png")

def update_iou_stats(prediction, ground_truth, intersections, unions):
    """累计单张预测图的每类 intersection / union。"""
    valid_mask = (ground_truth >= 0) & (ground_truth < NUM_CLASSES)

    for class_id in range(NUM_CLASSES):
        pred_mask = (prediction == class_id) & valid_mask
        gt_mask = (ground_truth == class_id) & valid_mask

        intersections[class_id] += np.logical_and(pred_mask, gt_mask).sum()
        unions[class_id] += np.logical_or(pred_mask, gt_mask).sum()

def print_miou(intersections, unions, samples_processed):
    ious = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    valid_classes = unions > 0
    ious[valid_classes] = intersections[valid_classes] / unions[valid_classes]
    miou = np.nanmean(ious) if np.any(valid_classes) else float('nan')

    print("\n========== Visualization Samples IoU ==========")
    print(f"Samples evaluated: {samples_processed}")
    for class_id in range(NUM_CLASSES):
        class_name = UDD_CLASS_NAMES[class_id]
        if np.isnan(ious[class_id]):
            print(f"{class_id}: {class_name:<10} IoU: nan")
        else:
            print(f"{class_id}: {class_name:<10} IoU: {ious[class_id]:.4f}")
    print(f"mIoU: {miou:.4f}")
    print("==============================================\n")

def predict_and_visualize(args, model, test_loader, num_samples=5):
    """
    对测试集进行预测并可视化结果
    
    Args:
        args: 命令行参数
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_samples: 要可视化的样本数量
    """
    model.eval()
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    
    samples_processed = 0
    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions = np.zeros(NUM_CLASSES, dtype=np.float64)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if samples_processed >= num_samples:
                break
                
            # 根据数据集格式获取数据
            if args.dataset == 'udd':
                input_tensor, label_tensor = batch
            else:
                input_tensor, label_tensor = batch
                size = None
                name = [f'sample_{i}']
            
            input_var = input_tensor.cuda()
            input_var = input_var.unsqueeze(0)
            
            # 进行预测
            output = model(input_var)
            output = output.squeeze(0)  # (N, C, H, W)
            predictions = torch.argmax(output, dim=1).cpu().numpy()
            
            # 充值神经元状态
            functional.reset_net(model)

            batch_size = predictions.shape[0]
            for batch_idx in range(batch_size):
                if samples_processed >= num_samples:
                    break

                prediction = predictions[batch_idx]

                # 获取原始图像（反标准化）
                image = input_tensor[batch_idx].cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
                
                # 获取真实标签
                ground_truth = label_tensor[batch_idx].cpu().numpy() if label_tensor is not None else None

                if ground_truth is not None:
                    update_iou_stats(prediction, ground_truth, intersections, unions)
                
                # 可视化
                save_path = os.path.join(save_dir, f'figure_{samples_processed}_visualization.png')
                visualize_prediction(image, prediction, ground_truth, save_path=save_path, show=False)
                
                # print(f"Processed sample {samples_processed + 1}: {name[0]}")
                samples_processed += 1
    
    print(f"Visualized {samples_processed} samples. Results saved to {save_dir}")
    print_miou(intersections, unions, samples_processed)
    create_legend()

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize UDD prediction results')
    parser.add_argument('--model', default="SpikingLETNet_shallow_middle", help="model name")
    parser.add_argument('--dataset', default="udd", help="dataset name")
    parser.add_argument('--checkpoint', type=str, required=True, help="path to model checkpoint")
    parser.add_argument('--save_dir', type=str, default="./udd_visualization", help="directory to save visualizations")
    parser.add_argument('--name', type=str, default="visualization", help="name prefix for saved visualizations")
    parser.add_argument('--num_samples', type=int, default=10, help="number of samples to visualize")
    parser.add_argument('--num_workers', type=int, default=2, help="number of data loading workers")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--cuda', default=True, help="use GPU")
    parser.add_argument('--gpus', default="0", type=str, help="GPU ids")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # # 设置GPU
    # if args.cuda:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    #     if not torch.cuda.is_available():
    #         raise Exception("No GPU found")
    device = torch.device(f"cuda:{args.gpus}" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # 构建模型
    model = build_model(args.model, num_classes=6, config="Network/configs/SpikingLETNet_shallow/1.3M.yaml")  # UDD有6个类别
    functional.set_step_mode(model, step_mode='m')
    param_num = sum(p.numel() for p in model.parameters())
    print(f"total_params: {param_num}")
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        if args.checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
        elif args.checkpoint.endswith('.pt'):
            model = torch.load(args.checkpoint, weights_only=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    if args.cuda:
        model = model.cuda()
        cudnn.benchmark = True

    model_q = quantize_model(model, k=4, inplace=False, quant=True, activation_quant=True, quant_start_layer=1, activation_quant_mode='per_channel')

    # 切换计算模式
    set_computing_mode(model_q, mode='pytorch')
    set_T_step(model_q, T=8)
    # for m in model.modules():
    #     if isinstance(m, QIFNode):
    #         m.T = 8
    # 加载测试集
    _, test_loader = build_dataset_test(args.dataset, args.num_workers, batch_size=args.batch_size)


    print(f"Starting visualization of {args.num_samples} samples...")
    predict_and_visualize(args, model_q, test_loader, args.num_samples)

if __name__ == '__main__':
    main()
