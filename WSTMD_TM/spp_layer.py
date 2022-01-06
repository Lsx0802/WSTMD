import math
import torch.nn as nn
import torch
import torch.nn.functional as F

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = max(math.ceil(previous_conv_size[0] / out_pool_size[i]),1)
        w_wid = max(math.ceil(previous_conv_size[1] / out_pool_size[i]),1)
        h_pad = min(math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2),math.floor(h_wid/2))
        w_pad = min(math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2),math.floor(w_wid/2))

        #print([h_wid,w_wid,h_pad,w_pad])
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

def SPPLayer(pool_type,x,num_levels):
    pool_type = pool_type
    num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
    for i in range(num_levels):
        level = i + 1
        kernel_size = (math.ceil(h / level), math.ceil(w / level))
        stride = (math.ceil(h / level), math.ceil(w / level))
        pooling = (
        math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

        # 选择池化方式
        if pool_type == 'max_pool':
            tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
        else:
            tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

        # 展开、拼接
        if (i == 0):
            x_flatten = tensor.view(num, -1)
        else:
            x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

    return x_flatten