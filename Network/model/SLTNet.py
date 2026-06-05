import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from model.module.transformer import TransBlock
# from model.module.patch import reverse_patches
from .module.SDSA import MS_Block
from spikingjelly.activation_based import layer, neuron, functional
from timm.models.layers import trunc_normal_

__all__ = ["SLTNet"]


class Spike_Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = layer.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            # self.bn_prelu = BNPReLU(nOut)
            self.bn_lif = BNLIF(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            # output = self.bn_prelu(output)
            output = self.bn_lif(output)

        return output



class BNLIF(nn.Module):
    def __init__(self, nIn, lif=True):
        super().__init__()
        self.bn = layer.BatchNorm2d(nIn, eps=1e-3)
        self.lif = neuron.LIFNode()
        self.lif_acti = lif

    def forward(self, input):
        output = self.bn(input)
        
        if self.lif_acti:
            output = self.lif(output)

        return output



class BasicInterpolate(nn.Module):
    def __init__(self, size, mode, align_corners):
        super(BasicInterpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode=self.mode,
                          align_corners=self.align_corners)
        return x


class Spike_LMSBModule(nn.Module):
     # 输出是membrane potential
    def __init__(self, nIn, d=1, kSize=3, dkSize=3, groups=1):
        super().__init__()

        self.ca_groups = groups
        self.bn_lif_1 = BNLIF(nIn)
        self.conv1x1_in = Spike_Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Spike_Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Spike_Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = Spike_Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Spike_Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)
        # self.spike_ca1 = Spike_CA(nIn // 2, groups = self.ca_groups)
        
        self.ddconv3x1 = Spike_Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Spike_Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)
        # self.spike_ca2 = Spike_CA(nIn // 2, groups = self.ca_groups)

        self.bn_lif_2 = BNLIF(nIn // 2)
        self.conv1x1 = Spike_Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)
        
        self.conv3x3_1 = Spike_Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, dilation=1, groups=nIn // 2, bn_acti=True)
        self.conv3x3_2 = Spike_Conv(nIn // 2, nIn // 2, dkSize, 1, padding=1, dilation=1, groups=nIn // 2, bn_acti=True)
        # self.conv3x3_3 = Spike_Conv(nIn // 2, nIn // 2, kernel_size=dkSize, 1, padding=d, dilation=d, groups=nIn // 2, bn_acti=True)
        
    def forward(self, input):
        output = self.bn_lif_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_lif_2(output)
        output = self.conv1x1(output)
        # output = self.shuffle(output + input)
        output = output + input

        return output
    

class Inverted_Spike_LMSBModule(nn.Module):
     # 输出是membrane potential
    def __init__(self, nIn, d=1, kSize=3, dkSize=3, groups=1):
        super().__init__()
        
        self.ca_groups = groups
        self.bn_lif_1 = BNLIF(nIn)
        self.conv1x1_in = Spike_Conv(nIn, nIn *6, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Spike_Conv(nIn *6, nIn *6, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Spike_Conv(nIn *6, nIn *6, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = Spike_Conv(nIn *6, nIn *6, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Spike_Conv(nIn *6, nIn *6, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn *6)
        # self.spike_ca1 = Spike_CA(nIn *6, groups = self.ca_groups)
        
        self.ddconv3x1 = Spike_Conv(nIn *6, nIn *6, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Spike_Conv(nIn *6, nIn *6, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn *6)
        # self.spike_ca2 = Spike_CA(nIn *6, groups = self.ca_groups)

        self.bn_lif_2 = BNLIF(nIn *6)
        self.conv1x1 = Spike_Conv(nIn *6, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn)
        
    def forward(self, input):
        output = self.bn_lif_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_lif_2(output)
        output = self.conv1x1(output)
        # output = self.shuffle(output + input)
        output = output + input

        return output



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [T,N,C,H,W] -> [T,N,g,C/g,H,W] -> [T,N,C/g,g,H,w] -> [T,N,C,H,W]'''
        T, N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(T, N, g, int(C / g), H, W).permute(0, 1, 3, 2, 4, 5).contiguous().view(T, N, C, H, W)
    

class DownSamplingBlock(nn.Module):
    # 输出是 membrane potential
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Spike_Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = layer.MaxPool2d(2, stride=2)
        self.bnlif = BNLIF(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        # output = self.bnlif(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = layer.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = layer.BatchNorm2d(noutput, eps=1e-3)
        # self.relu = nn.ReLU6(inplace=True) # ReLU6 的输出范围是[0,6],主要在移动和嵌入式设备上的神经网络中使用，因为它在数值范围上的限制（最大值为6）有助于量化和硬件加速
        self.lif = neuron.LIFNode()

    def forward(self, input):
        output = self.conv(input)
        # output = self.bn(output)
        # output = self.lif(output)
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

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [t,b,c,1,1]

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        # self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        # self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        self.dconv3x1 = Spike_Conv(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.dconv1x3 = Spike_Conv(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1), bn_acti=True)
        
        if self.bn_acti:
            self.bn_lif = BNLIF(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_lif(output)

        return output
    


class FeatureEnhance(nn.Module):
    def __init__(self, nIn, r):
        super().__init__()

        self.GAP = layer.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = Spike_Conv(nIn, nIn//r, kSize=1, stride=1, padding=0, bn_acti=True)
        self.conv1x1_2 = Spike_Conv(nIn//r, nIn, kSize=1, stride=1, padding=0, bn_acti=False)
        
        # b,c,h,w
        self.conv3x3 = Spike_Conv(2, 1, kSize=3, stride=1, padding=1, bn_acti=False)
        self.bn = layer.BatchNorm2d(1, eps=1e-3)
        
        self.conv1x1 = Spike_Conv(nIn, nIn, kSize=1, stride=1, padding=0, bn_acti=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # channel
        output1 = self.GAP(input)
        output1 = self.conv1x1_1(output1)
        output1 = self.sigmoid(self.conv1x1_2(output1))
        output1 = input * output1
        
        # spatial
        mean = torch.mean(input, dim=1, keepdim=True)
        max, _ = torch.max(input, dim=1, keepdim=True)
        output2 = torch.cat((mean, max), dim=1)
        output2 = self.sigmoid(self.bn(self.conv3x3(output2)))
        output2 = input * output2
        
        # fuse
        output = input + self.conv1x1(output1) + self.conv1x1(output2)
        
        return output
                 

class SLTNet(nn.Module):
    # def __init__(self, classes=19, block_1=3, block_2=12, block_3=12, block_4=3, block_5 = 3, block_6 = 3, ohem=False, augment=False):
    def __init__(self, classes=19, block_1=3, block_2=3, block_3=3, block_4=1, block_5=1, block_6=1, ohem=False, augment=False):
        super().__init__()
        self.augment = augment
        self.ohem = ohem
        
        self.init_conv = nn.Sequential(
            # Spike_Conv(1, 32, 3, 1, padding=1, bn_acti=True),
            Spike_Conv(3, 32, 3, 1, padding=1, bn_acti=True),
            Spike_Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Spike_Conv(32, 32, 3, 2, padding=1, bn_acti=False),
        )

        # self.bn_prelu_1 = BNPReLU(32)
        self.bn_lif_1 = BNLIF(32)

        self.downsample_1 = DownSamplingBlock(32, 64)

        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), Spike_LMSBModule(64, d=2))
        self.bn_lif_2 = BNLIF(64)

        # DAB Block 2
        dilation_block_2 = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.downsample_2 = DownSamplingBlock(64, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        Spike_LMSBModule(128, d=dilation_block_2[i], groups=1))
        self.bn_lif_3 = BNLIF(128)

        # DAB Block 3
        #dilation_block_3 = [2, 5, 7, 9, 13, 17]
        dilation_block_3 = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.downsample_3 = DownSamplingBlock(128, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        Spike_LMSBModule(32, d=dilation_block_3[i]))
        self.bn_lif_4 = BNLIF(32)
        
        
        
        self.transformer1 = MS_Block(dim=32, hid_dim=32, num_heads=2)
        self.transformer2 = MS_Block(dim=32, hid_dim=32, num_heads=2)
        
        # self.transformer1 = Block(dim=32, num_heads=2)
        # self.transformer2 = Block(dim=32, num_heads=2)
        
        
#DECODER
        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
           self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                       Spike_LMSBModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 16)
        self.bn_lif_5 = BNLIF(16)
        

        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        Spike_LMSBModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.bn_lif_6 = BNLIF(16)
        
        
        dilation_block_6 = [2, 2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        Spike_LMSBModule(16, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(16, 16)
        self.bn_lif_7 = BNLIF(16)
        
        
        self.PA3 = PA(nf=16)


        self.LC1 = LongConnection(nIn=64, nOut=16, kSize=3, bn_acti=False)
        self.LC2 = LongConnection(nIn=128, nOut=16, kSize=3, bn_acti=False)
        # self.LC3 = LongConnection(32, 16, 3)
        self.LC3 = LongConnection(nIn=32, nOut=32, kSize=3, bn_acti=False)


        self.FE1 = FeatureEnhance(nIn=16, r=2)
        self.FE2 = FeatureEnhance(nIn=16, r=2)
        self.FE3 = FeatureEnhance(nIn=32, r=2)

        
        self.classifier = nn.Sequential(Spike_Conv(16, classes, 1, 1, padding=0))
        
        self.apply(self.trunc_init)
        functional.set_step_mode(self, step_mode='s')
        # self.apply(self.kaiming_init)

    
    # 截断正态分布初始化：从一个正态分布（u，σ）中抽取样本，然后在（u-kσ，u+kσ）范围内抽样，k是超参数
    # 服从标准的正态分布初始化，权重在具有特定均值和方差的正态分布中随机抽取，然而，正态分布的长尾性质，可能会抽到极大值或极小值，从而导致激活函数饱和（梯度消失），从而影响模型训练结果
    # 截断正态分布通过限制权重的抽取范围来解决这个问题，超过该范围的权重将被截断，然后重新抽取新的值，直到所有权重被填充 
    def trunc_init(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

            

    def kaiming_init(self, m):
        if isinstance(m, (nn.Conv2d, layer.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
                

    def forward(self, input):

        # [b,t,c,h,w]->[t,b,c,h,w]
        # input = input.permute(1, 0, 2, 3, 4).contiguous()

        output0 = self.init_conv(input)
        output0 = self.bn_lif_1(output0)

        # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        output1 = self.bn_lif_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_lif_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_lif_4(output3)

#Transformer

        output4_ = self.transformer1(output3)
        output4 = self.transformer2(output4_)
        
        # output4 = output4.permute(0, 2, 1)
        # output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)
        
#DECODER            
        output4 = self.DAB_Block_4(output4)
        # output4:[8, 32, 23, 30], output3:[8, 32, 23, 30], temp:[8, 16, 23, 30]
        
        output4 = self.upsample_1(output4 + self.FE3(self.LC3(output3)))
        
        output4 = self.bn_lif_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        temp = self.FE2(self.LC2(output2))
        output5 = BasicInterpolate(size=temp.size()[2:], mode='bilinear', align_corners=False)(output5)
        output5 = self.upsample_2(output5 + temp)
        
        output5 = self.bn_lif_6(output5)
        
        if self.augment:
            early_out = BasicInterpolate(size=input.size()[2:], mode='bilinear', align_corners=False)(output5)
            early_out = self.classifier(early_out)
        
        output6 = self.DAB_Block_6(output5)
        output6 = self.upsample_3(output6 + self.FE1(self.LC1(output1)))
        output6 = self.PA3(output6)
        # output6 = self.bn_lif_7(output6) 删除是因为担心0/1的插值会很大影响精度
        
        
        out = BasicInterpolate(size=input.size()[2:], mode='bilinear', align_corners=False)(output6)
        out = self.bn_lif_7(out) # 挪动了位置，想要在输入分类器前变为0/1

        out = self.classifier(out)

        # 在T维上取平均：[T,B,C,H,W]->[B,C,H,W]
        
        if self.augment & self.ohem :
            return [out, early_out]
        elif self.augment:
            return early_out
        else:  
            return out
        

        
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

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cpu")
    model = SLTNet(classes=19).to(device)
    summary(model, (3, 512, 1024), device="cpu")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

