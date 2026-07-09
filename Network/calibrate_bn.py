import copy
import os
import time
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn



SUPPORTED_MODELS = (
    "SpikingLETNet_shallow_max",
    "SpikingLETNet_shallow_middle",
    "SpikingLETNet_shallow_small",
)


def parse_args():
    parser = ArgumentParser(
        description="Calibrate BatchNorm running statistics for SpikingLETNet shallow variants."
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-max", type=str, default=None)
    parser.add_argument("--checkpoint-middle", type=str, default=None)
    parser.add_argument("--checkpoint-small", type=str, default=None)
    parser.add_argument("--config", type=str, default="Network/configs/SpikingLETNet_shallow/1.3M.yaml")
    parser.add_argument("--dataset", type=str, default="udd", choices=["udd", "voc", "cityscapes", "camvid"])
    parser.add_argument("--input_size", type=str, default="400,400")
    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--train_type", type=str, default="trainval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--max_batches", type=int, default=200, help="0 means use the whole calibration loader")
    parser.add_argument("--T", type=int, default=1, help="timesteps used by the spiking model")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="which loader returned by build_dataset_train is used for calibration",
    )
    parser.add_argument(
        "--keep-bn-stats",
        action="store_true",
        help="keep existing running stats before calibration instead of resetting them",
    )
    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=None,
        help="BN momentum during calibration. Default None uses cumulative moving average.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_bn_calibrated",
        help="suffix inserted before .pth when saving beside the original checkpoint",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="save back to the original checkpoint path instead of creating a suffixed file",
    )
    parser.add_argument(
        "--allow-non-bn-mismatch",
        action="store_true",
        help="allow missing or shape-mismatched non-BN tensors when loading the checkpoint",
    )
    return parser.parse_args()


def infer_classes(dataset, classes):
    if classes is not None:
        return classes
    if dataset == "cityscapes":
        return 19
    if dataset == "camvid":
        return 11
    if dataset == "voc":
        return 21
    if dataset == "udd":
        return 6
    raise NotImplementedError(f"Unsupported dataset: {dataset}")


def collect_jobs(args):
    jobs = []
    if args.model is not None or args.checkpoint is not None:
        if args.model is None or args.checkpoint is None:
            raise ValueError("Use --model and --checkpoint together for single-model calibration.")
        jobs.append((args.model, args.checkpoint))

    named_checkpoints = (
        ("SpikingLETNet_shallow_max", args.checkpoint_max),
        ("SpikingLETNet_shallow_middle", args.checkpoint_middle),
        ("SpikingLETNet_shallow_small", args.checkpoint_small),
    )
    jobs.extend((model_name, path) for model_name, path in named_checkpoints if path)

    if not jobs:
        raise ValueError(
            "No checkpoint specified. Use --model/--checkpoint or one of "
            "--checkpoint-max, --checkpoint-middle, --checkpoint-small."
        )
    return jobs


def is_bn_module(module):
    return isinstance(module, nn.modules.batchnorm._BatchNorm)


def get_bn_modules(model):
    return [(name, module) for name, module in model.named_modules() if is_bn_module(module)]


def key_belongs_to_bn(key, bn_names):
    return any(key == name or key.startswith(f"{name}.") for name in bn_names)


def reset_bn_stats(bn_modules):
    for _, module in bn_modules:
        module.reset_running_stats()


def set_bn_calibration_mode(model, bn_modules, momentum):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, module in bn_modules:
        module.train()
        module.momentum = momentum


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_compatible_state_dict(model, checkpoint_state):
    model_state = model.state_dict()
    compatible_state = {}
    unexpected_keys = []
    shape_mismatch = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    merged_state = copy.copy(model_state)
    merged_state.update(compatible_state)
    load_result = model.load_state_dict(merged_state, strict=True)

    loaded_keys = set(compatible_state)
    missing_keys = [key for key in model_state if key not in loaded_keys]
    return load_result, missing_keys, unexpected_keys, shape_mismatch


def make_output_path(checkpoint_path, suffix, overwrite):
    if overwrite:
        return checkpoint_path
    dirname, basename = os.path.split(checkpoint_path)
    stem, ext = os.path.splitext(basename)
    if not ext:
        ext = ".pth"
    return os.path.join(dirname, f"{stem}{suffix}{ext}")


def unpack_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def calibrate_one(args, model_name, checkpoint_path, train_loader, val_loader, device):
    from builders.model_builder import build_model
    from spikingjelly.activation_based import functional

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"=====> building {model_name}")
    model = build_model(model_name, num_classes=args.classes, config=args.config)
    functional.set_step_mode(model, step_mode="m")

    print(f"=====> loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = extract_state_dict(checkpoint)
    _, missing_keys, unexpected_keys, shape_mismatch = load_compatible_state_dict(model, checkpoint_state)

    print(f"Loaded tensors: {len(checkpoint_state) - len(unexpected_keys) - len(shape_mismatch)}")
    print(f"Missing/current-initialized tensors: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected checkpoint tensors skipped: {len(unexpected_keys)}")
    if shape_mismatch:
        print(f"Shape-mismatched tensors skipped: {len(shape_mismatch)}")
        for key, ckpt_shape, model_shape in shape_mismatch[:20]:
            print(f"  {key}: checkpoint{ckpt_shape} -> model{model_shape}")
        if len(shape_mismatch) > 20:
            print(f"  ... {len(shape_mismatch) - 20} more")

    bn_modules = get_bn_modules(model)
    if not bn_modules:
        raise RuntimeError("No BatchNorm modules found in model.")
    print(f"Found BatchNorm layers: {len(bn_modules)}")

    bn_names = [name for name, _ in bn_modules]
    non_bn_missing = [key for key in missing_keys if not key_belongs_to_bn(key, bn_names)]
    non_bn_unexpected = [key for key in unexpected_keys if not key_belongs_to_bn(key, bn_names)]
    non_bn_shape_mismatch = [item for item in shape_mismatch if not key_belongs_to_bn(item[0], bn_names)]
    if not args.allow_non_bn_mismatch and (non_bn_missing or non_bn_unexpected or non_bn_shape_mismatch):
        raise RuntimeError(
            "Checkpoint mismatch contains non-BN tensors. "
            "Check that --model and --config match the checkpoint, or rerun with "
            "--allow-non-bn-mismatch if this is intentional.\n"
            f"non-BN missing: {non_bn_missing[:20]}\n"
            f"non-BN unexpected: {non_bn_unexpected[:20]}\n"
            f"non-BN shape mismatch: {non_bn_shape_mismatch[:20]}"
        )

    if not args.keep_bn_stats:
        reset_bn_stats(bn_modules)
        print("Reset BN running stats before calibration.")

    set_bn_calibration_mode(model, bn_modules, args.bn_momentum)
    model.to(device)
    cudnn.benchmark = not args.cpu

    loader = train_loader if args.split == "train" else val_loader
    max_batches = len(loader) if args.max_batches == 0 else min(args.max_batches, len(loader))
    print(f"=====> calibrating on {args.split} loader for {max_batches} batches")

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = unpack_images(batch).to(device, non_blocking=True)
            images = images.repeat(args.T, 1, 1, 1, 1)
            model(images)
            functional.reset_net(model)

            if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == max_batches:
                print(f"  calibrated {batch_idx + 1}/{max_batches} batches")

    elapsed = time.time() - start_time
    print(f"BN calibration finished in {elapsed:.1f}s")

    model.eval()
    calibrated_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    output_path = make_output_path(checkpoint_path, args.output_suffix, args.overwrite)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        output_checkpoint = dict(checkpoint)
        output_checkpoint["model"] = calibrated_state
        output_checkpoint["bn_calibration"] = {
            "model": model_name,
            "config": args.config,
            "dataset": args.dataset,
            "split": args.split,
            "max_batches": max_batches,
            "T": args.T,
            "reset_bn_stats": not args.keep_bn_stats,
            "bn_momentum": args.bn_momentum,
        }
    else:
        output_checkpoint = calibrated_state

    torch.save(output_checkpoint, output_path)
    print(f"Saved calibrated checkpoint: {output_path}")
    return output_path


def main():
    args = parse_args()
    args.classes = infer_classes(args.dataset, args.classes)
    jobs = collect_jobs(args)

    h, w = map(int, args.input_size.split(","))
    input_size = (h, w)
    print(f"=====> input size: {input_size}")
    print(f"=====> classes: {args.classes}")

    if args.cpu:
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --cpu to run on CPU.")
        torch.cuda.set_device(0)
        device = torch.device("cuda")
    print(f"=====> calibration device: {device}")

    from builders.dataset_builder import build_dataset_train

    _, train_loader, val_loader = build_dataset_train(
        args.dataset,
        input_size,
        args.batch_size,
        args.train_type,
        random_scale=False,
        random_mirror=False,
        num_workers=args.num_workers,
    )

    saved_paths = []
    for model_name, checkpoint_path in jobs:
        saved_paths.append(calibrate_one(args, model_name, checkpoint_path, train_loader, val_loader, device))

    print("=====> all done")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
