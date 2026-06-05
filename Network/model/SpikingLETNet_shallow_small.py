import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import yaml
# from IPython import embed
from .module.transformer import TransBlock, STransBlock
from .module.patch import reverse_patches
from .module.SDSA import MS_Block
from .module.neuron import QIFNode
from spikingjelly.activation_based import neuron, functional, layer

__all__ = ["SpikingLETNet"]


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = layer.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = layer.BatchNorm2d(nIn, eps=1e-3)
        self.acti = QIFNode()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        # self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=False)
        self.ca11 = eca_layer(nIn // 2)
        
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=False)
        self.ca22 = eca_layer(nIn // 2)

        # self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)
        self.bn_relu = BNPReLU(nIn)
        
    def forward(self, input):
        # output = self.bn_relu_1(input)
        output = self.conv1x1_in(input)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        # output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        output = self.shuffle(output + input)
        output = self.bn_relu(output)

        return output

        #return output + input



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        T, N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(T, N, g, int(C / g), H, W).permute(0, 1, 3, 2, 4, 5).contiguous().view(T, N, C, H, W)
    
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
        self.conv = layer.Conv2d(nIn, nIn, kernel_size=3, padding=1)
        self.avg_pool = layer.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        input = self.conv(input)
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            avg_pool = self.avg_pool(input)
            output = torch.cat([output, avg_pool], 2)

        output = self.bn_prelu(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = layer.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = layer.BatchNorm2d(noutput, eps=1e-3)
        self.relu = QIFNode()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output
        
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = layer.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = layer.AdaptiveAvgPool2d(1)
        self.conv = layer.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.neuron = QIFNode()

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        t = x * y.expand_as(x)
        t = self.neuron(t)
        return t


        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize,  bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = layer.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = layer.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
                 

def set_qif_T(module: nn.Module, T: int):
    """
    递归地遍历所有子模块并为所有 QIFNode 实例设置 T。
    """
    for m in module.modules():
        if isinstance(m, QIFNode):
            m.T = T


class SpikingLETNet_shallow_small(nn.Module):
    def __init__(self, classes=19, config=None):
        super().__init__()
        assert config is not None, "Please provide a config dictionary."
        with open(config, "r") as f:
            config = yaml.safe_load(f)
        block_1 = config['model']['block_1']
        block_2 = config['model']['block_2']
        block_3 = config['model']['block_3']
        block_4 = config['model']['block_4']
        block_5 = config['model']['block_5']
        block_6 = config['model']['block_6']
        self.T = config['model']['T']

        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            # Conv(64, 64, 3, 2, padding=1, bn_acti=True),
        )

        # self.bn_prelu_1 = BNPReLU(32)

        self.downsample_1 = DownSamplingBlock(32, 64)

        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(64)

        # DAB Block 2
        dilation_block_2 = [1, 2, 4]
        self.downsample_2 = DownSamplingBlock(64, 64)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(64, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(64)

        # DAB Block 3
        #dilation_block_3 = [2, 5, 7, 9, 13, 17]
        dilation_block_3 = [1, 2, 4]
        self.downsample_3 = DownSamplingBlock(64, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(32, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(32)
        
        

        self.transformer1 = STransBlock(dim=288)
        # self.transformer1 = MS_Block(dim=64, hid_dim=128, num_heads=4)
        
        
#DECODER
        dilation_block_4 = [2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
           self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                       DABModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 32)
        self.bn_prelu_5 = BNPReLU(32)
        

        dilation_block_5 = [2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        DABModule(32, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(32, 32)
        self.bn_prelu_6 = BNPReLU(32)
        
        
        dilation_block_6 = [2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        DABModule(32, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(32, 32)
        self.bn_prelu_7 = BNPReLU(32)
        
        
        # self.PA1 = PA(16)
        # self.PA2 = PA(16)
        self.PA3 = PA(32)


        
        self.LC1 = LongConnection(64, 32, 3, bn_acti=True)
        self.LC2 = LongConnection(64, 32, 3, bn_acti=True)
        self.LC3 = LongConnection(32, 32, 3, bn_acti=True)

        self.final_conv = layer.Conv2d(32, 32, 3, 1, 1)
        
        self.classifier = nn.Sequential(Conv(32, classes, 1, 1, padding=0))

        set_qif_T(self, self.T)

        functional.set_step_mode(self, step_mode='s')

    def forward(self, input):

        output0 = self.init_conv(input)
        # output0 = self.bn_prelu_1(output0)

        # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        # output1 = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        # output2 = self.bn_prelu_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        # output3 = self.bn_prelu_4(output3)

#Transformer

        t, b, c, h, w = output3.shape
        output3 = output3.mean(0)
        output4 = self.transformer1(output3)
        
        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)
        output4 = output4.repeat(t, 1, 1, 1, 1)
        
#DECODER            
        output4 = self.DAB_Block_4(output4)
        output4 = self.upsample_1(output4 + self.LC3(output3.unsqueeze(0)))
        
        # output4 = self.bn_prelu_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        output5 = self.upsample_2(output5 + self.LC2(output2))
        
        # output5 = self.bn_prelu_6(output5)
        
        
        output6 = self.DAB_Block_6(output5)
        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.final_conv(output6)
        # output6 = self.PA3(output6)
        # output6 = self.bn_prelu_7(output6)
        
        y_shape = [output6.shape[0], output6.shape[1]]
        output6 = output6.flatten(0, 1)
        out = F.interpolate(output6, input.size()[3:], mode='bilinear', align_corners=False)
        y_shape.extend(out.shape[1:])
        out = out.view(y_shape)
        out = self.classifier(out)
        return out
    
    def forward_qat(self, input):

        output0 = self.init_conv(input)
        # output0 = self.bn_prelu_1(output0)

        # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        # output1 = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        # output2 = self.bn_prelu_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        # output3 = self.bn_prelu_4(output3)

#Transformer

        t, b, c, h, w = output3.shape
        output3 = output3.mean(0)
        output4 = self.transformer1(output3)
        
        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)
        output4 = output4.repeat(t, 1, 1, 1, 1)
        
#DECODER            
        output4 = self.DAB_Block_4(output4)
        output4 = self.upsample_1(output4 + self.LC3(output3.unsqueeze(0)))
        
        # output4 = self.bn_prelu_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        output5 = self.upsample_2(output5 + self.LC2(output2))
        
        # output5 = self.bn_prelu_6(output5)
        
        
        output6 = self.DAB_Block_6(output5)
        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.final_conv(output6)
        # output6 = self.PA3(output6)
        # output6 = self.bn_prelu_7(output6)
        
        y_shape = [output6.shape[0], output6.shape[1]]
        output6 = output6.flatten(0, 1)
        out = F.interpolate(output6, input.size()[3:], mode='bilinear', align_corners=False)
        y_shape.extend(out.shape[1:])
        out = out.view(y_shape)
        out = self.classifier(out)

        mid_outputs = [output1, output2, output3, output4, output5, output6]

        return out, mid_outputs


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cpu")
    model = SpikingLETNet_shallow(classes=19, config='/storage/lmh/LETNet/Network/configs/SpikingLETNet_shallow/1.3M.yaml').to(device)
    # total_paramters = netParams(model)
    # print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
    functional.set_step_mode(model, step_mode='m')
    # summary(model, (4, 3, 400, 400), batch_size=1)
    i = torch.randn(1, 4, 3, 400, 400).to(device)
    model(i)