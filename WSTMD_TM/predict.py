# coding: utf-8
import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from data_pre2 import box11
from model import WSDDN_res
import cv2
import numpy as np
from tqdm import tqdm


def pre(img, label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    box1 = box11(img)
    box=box1
    # print(box)
    ssw_block = torch.Tensor(int((len(box)) / 4), 4)

    for i in range(int((len(box)) / 4)):
        ssw_block[i, 0] = round(float(box[i * 4 + 0]) / 32)
        ssw_block[i, 1] = round(float(box[i * 4 + 1]) / 32)
        ssw_block[i, 2] = round(float(box[i * 4 + 2]) / 32)
        ssw_block[i, 3] = round(float(box[i * 4 + 3]) / 32)

    img = Image.open(img)

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = WSDDN_res().to(device)

    # load model weights
    weights_path = "Resnet34_WSDDN.pkl"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    with torch.no_grad():
        # predict class
        img = img.to(device)
        box = torch.unsqueeze(ssw_block, dim=0)
        box = box.to(device)
        val_outputs, val_op2, val_op3= model(img, box)

        # f = open('box_result2.txt', 'w')
        box_2=[]
        for j in range(val_outputs.size(0)):
            for k in range(val_op2.size(1)):
                if val_op3[j, k, 1] >(1/val_op3.size(1)) or val_op3[j, k, 0] >(1/val_op3.size(1)):
                    new_line = [float('%.3f' % val_op2[j, k, 1].item()),
                                box1[4*k+0],
                                box1[4*k+1],  box1[4*k+2],
                                 box1[4*k+3]]
                    box_2.append(new_line)


        val_output = torch.squeeze(val_outputs).cpu()
        predict = torch.softmax(val_output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "original:{} predict: {}  prob: {:.4}".format(label, class_indict[str(predict_cla)],
                                                              predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)

    return box_2


def plot_box(img, box):
    if box[0] > 0.5:
        label = 'TM'
    else:
        label = 'NTM'
    img=cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[1] + box[3]), int(box[2] + box[4])),color=(255, 255, 0), thickness=1)
    plt.imshow(img)
    img=cv2.putText(img, label, (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 255, 0), thickness=1)
    plt.imshow(img)

def main(fold):
    save_path = './final_img2'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    path = 'data/img/'
    fh = open('txt/val'+fold+'_t.txt', 'r')
    # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
    for line in tqdm(fh):  # 迭代该列表#按行循环txt文本中的内
        line = line.strip('\n')
        line = line.rstrip('\n')
        # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
        words = line.split()
        img_path = os.path.join(path, words[0])
        img = img_path
        if words[1] == '0':
            label = 'NTM'
        else:
            label = 'TM'

        box_2 = pre(img, label)

        img_name = img_path
        img_name = cv2.imread(img_name)
        img_name = cv2.resize(img_name, (224, 224))
        img_name = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)

        for box in box_2:
            plot_box(img_name, box)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, words[0]), bbox_inches='tight',pad_inches = 0)

def single(path,label):

    img = path

    box_2 = pre(img, label)

    img_name = path
    img_name = cv2.imread(img_name)
    img_name = cv2.resize(img_name, (224, 224))
    img_name = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)

    for box in box_2:
        plot_box(img_name, box)
    plt.axis('off')
    plt.show()
    # plt.savefig(os.path.join(save_path, words[0]), bbox_inches='tight',pad_inches = 0)

if __name__ == '__main__':
    for fold in range(1,2):
        main(str(fold))
    # path = 'data/img/140829091953.png'
    # path='data/img/140828073135.png'
    # label = 'NTM'
    # single(path,label)
