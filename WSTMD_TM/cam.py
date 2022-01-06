
# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
from data_pre2 import box11
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import  WSDDN_res

IMAGE_SIZE=224

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)  # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    img = img[:, :, ::-1]  # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir,patient):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, patient+"cam.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))

def show_cam_on_image2(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    cv2.imwrite(path, np.uint8(255 * cam))

def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot.to(device) * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMAGE_SIZE,IMAGE_SIZE))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fmap_block = list()
    grad_block = list()

    path_img='data/img/140829091953.png'
    path_net="Resnet34_t_WSDDN.pkl"

    box1 = box11(path_img)
    box = box1
    # print(box)
    ssw_block = torch.Tensor(int((len(box)) / 4), 4)

    for i in range(int((len(box)) / 4)):
        ssw_block[i, 0] = round(float(box[i * 4 + 0]) / 32)
        ssw_block[i, 1] = round(float(box[i * 4 + 1]) / 32)
        ssw_block[i, 2] = round(float(box[i * 4 + 2]) / 32)
        ssw_block[i, 3] = round(float(box[i * 4 + 3]) / 32)

    box = torch.unsqueeze(ssw_block, dim=0)
    box = box.to(device)

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = WSDDN_res().to(device)
    net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.layer4.register_forward_hook(farward_hook)
    net.layer4.register_backward_hook(backward_hook)

    # forward
    output,op2,op3 = net(img_input.to(device),box)

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))) / 255
    show_cam_on_image2(img_show, cam, 'tongue_cam.jpg')








