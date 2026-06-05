from model.SQNet import SQNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.UNet import UNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
# from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.ContextNet import ContextNet
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FSSNet import FSSNet
from model.FPENet import FPENet
from model.LETNet import LETNet
from model.SpikingLETNet import SpikingLETNet
from model.SpikingLETNet_shallow import SpikingLETNet_shallow
from model.SpikingLETNet_shallow_small import SpikingLETNet_shallow_small
from model.SpikingLETNet_shallow_middle import SpikingLETNet_shallow_middle
from model.SpikingLETNet_shallow_max import SpikingLETNet_shallow_max
from model.SpikingLETNet_shallow_wo_Shuffle import SpikingLETNet_shallow_wo_Shuffle
from model.SLTNet import SLTNet
from model.SqueezeNet import SqueezeNet
from model.ShuffleNet import ShuffleNetV2
from torchvision.models.squeezenet import SqueezeNet1_1_Weights
from torchvision.models.shufflenetv2 import ShuffleNet_V2_X0_5_Weights
from model.MobileVit import MobileVit

def build_model(model_name, num_classes, config=None):
    if model_name == 'SQNet':
        return SQNet(classes=num_classes)
    elif model_name == 'LinkNet':
        return LinkNet(classes=num_classes)
    elif model_name == 'SegNet':
        return SegNet(classes=num_classes)
    elif model_name == 'UNet':
        return UNet(classes=num_classes)
    elif model_name == 'ENet':
        return ENet(classes=num_classes)
    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'ESNet':
        return ESNet(classes=num_classes)
    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)
    elif model_name == 'LEDNet':
        return LEDNet(classes=num_classes)
    elif model_name == 'ESPNet_v2':
        return EESPNet_Seg(classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)
    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'FSSNet':
        return FSSNet(classes=num_classes)
    elif model_name == 'FPENet':
        return FPENet(classes=num_classes)
    elif model_name == 'LETNet':
        return LETNet(classes=num_classes)
    elif model_name == 'SpikingLETNet':
        return SpikingLETNet(classes=num_classes)
    elif model_name == 'SLTNet':
        return SLTNet(classes=num_classes)
    elif model_name == 'SqueezeNet':
        model = SqueezeNet(version='1_1', num_classes=num_classes)
        model.load_state_dict(SqueezeNet1_1_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True), strict=False)
        return model
    elif model_name == 'ShuffleNet':
        model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=num_classes)
        model.load_state_dict(ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True), strict=False)
        return model
    elif model_name == 'MobileVit':
        model = MobileVit(num_classes=num_classes)
        return model
    elif model_name == 'SpikingLETNet_shallow':
        return SpikingLETNet_shallow(classes=num_classes, config=config)
    elif model_name == 'SpikingLETNet_shallow_small':
        return SpikingLETNet_shallow_small(classes=num_classes, config=config)
    elif model_name == 'SpikingLETNet_shallow_middle':
        return SpikingLETNet_shallow_middle(classes=num_classes, config=config)
    elif model_name == "SpikingLETNet_shallow_max":
        return SpikingLETNet_shallow_max(classes=num_classes, config=config)
    elif model_name == 'SpikingLETNet_shallow_wo_Shuffle':
        return SpikingLETNet_shallow_wo_Shuffle(classes=num_classes, config=config)
    else:
        raise Exception("the model name does not exist, please check the model name")

