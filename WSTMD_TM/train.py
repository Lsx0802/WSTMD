import os
import time

from model import resnet34, WSDDN_res
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import warnings
from MyDataset import MyDataset

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print("device is " + str(torch.cuda.get_device_name()))

def get_confusion_matrix(trues, preds):
    labels = [0, 1]
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix


def roc_auc(trues, preds):
    fpr, tpr, thresholds = roc_curve(trues, preds)
    auc = roc_auc_score(trues, preds)
    return fpr, tpr, auc


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def main():
    EPOCH = 300
    best_accuracy = 0.0
    trigger = 0
    early_stop_step = 20
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    IMAGE_HEIGHT, IMAGE_WIDTH = 224,224

    model_use_pretrain_weight = True
    image_path = 'data/img'  # tongue data set path
    train_txt_path = 'txt/train1_t.txt'
    val_txt_path = 'txt/val1_t.txt'
    ssw_path = 'ssw_5.txt'
    save_name = 'Resnet34_WSDDN'

    val_loss = []
    val_presision = []
    val_recall = []
    val_f1 = []
    val_accuracy = []
    val_conf_matrix = []

    model = WSDDN_res()

    if model_use_pretrain_weight:
        # model_weight_path = "resnet34-333f7ec4.pth"
        model_weight_path = "pre_weight/resnet34_t_pre_1.pkl"
        # model_weight_path = "resnet34_test.pkl"

        # model_weight_path = "vgg16-397923af.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

        pre_weights = torch.load(model_weight_path, map_location=device)
        del_key = []
        for key, _ in pre_weights.items():
            if "fc" in key:
                del_key.append(key)

        for key in del_key:
            del pre_weights[key]

        missing_keys, unexpected_keys = model.load_state_dict(pre_weights, strict=False)
        print("[missing_keys]:", *missing_keys, sep="\n")
        print("[unexpected_keys]:", *unexpected_keys, sep="\n")
        print("\n")

    print(model)
    print('params:' + str(count_params(model)) + '\n')

    model.to(device)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),

        "val": transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    train_dataset = MyDataset(data_dir=image_path, txt=train_txt_path, ssw_txt=ssw_path,
                              transform=data_transform['train'])
    val_dataset = MyDataset(data_dir=image_path, txt=val_txt_path, ssw_txt=ssw_path, transform=data_transform['val'])

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0)

    data_loader_val = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print("start training" + '\n')

    before = time.time()
    for epoch in range(EPOCH):
        tot_train_loss = 0.0
        tot_val_loss = 0.0

        train_preds = []
        train_trues = []

        model.train()
        for i, (train_data_batch, train_box_batch, train_label_batch) in tqdm(enumerate(data_loader_train),
                                                                              total=len(data_loader_train)):
            train_data_batch = train_data_batch.float().to(device)  # 将double数据转换为float
            train_label_batch = train_label_batch.to(device)
            train_box_batch = train_box_batch.float().to(device)

            train_outputs, train_op2, train_op3 = model(train_data_batch, train_box_batch)
            loss = criterion(train_outputs, train_label_batch)
            # print(loss)
            # 反向传播优化网络参数
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 累加每个step的损失
            tot_train_loss += loss.data
            train_outputs = train_outputs.argmax(dim=1)

            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(train_label_batch.detach().cpu().numpy())

        train_accuracy = accuracy_score(train_trues, train_preds)
        precision = precision_score(train_trues, train_preds)
        recall = recall_score(train_trues, train_preds)
        f1 = f1_score(train_trues, train_preds)

        print("[train] Epoch:{} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} loss:{:.4f}".format(
            epoch, train_accuracy, precision, recall, f1, tot_train_loss))

        val_preds = []
        val_trues = []

        model.eval()

        with torch.no_grad():
            for i, (val_data_batch, val_box_batch, val_label_batch) in tqdm(enumerate(data_loader_val),
                                                                            total=len(data_loader_val)):
                val_data_batch = val_data_batch.float().to(device)  # 将double数据转换为float
                val_label_batch = val_label_batch.to(device)
                val_box_batch = val_box_batch.float().to(device)
                val_outputs, val_op2, val_op3 = model(val_data_batch, val_box_batch)

                loss = criterion(val_outputs, val_label_batch)
                tot_val_loss += loss.data
                val_outputs = val_outputs.argmax(dim=1)

                val_preds.extend(val_outputs.detach().cpu().numpy())
                val_trues.extend(val_label_batch.detach().cpu().numpy())

            accuracy = accuracy_score(val_trues, val_preds)
            precision = precision_score(val_trues, val_preds)
            recall = recall_score(val_trues, val_preds)
            f1 = f1_score(val_trues, val_preds)
            conf_matrix = get_confusion_matrix(val_trues, val_preds)

            print("[val] Epoch:{} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} loss:{:.4f} ".format(epoch,
                                                                                                                accuracy,
                                                                                                                precision,
                                                                                                                recall,
                                                                                                                f1,
                                                                                                                tot_val_loss))

            trigger += 1
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "./" + save_name + '.pkl')
                print("save best weighted ")
                print("best_accuracy:{:.4f}".format(best_accuracy))
                # trigger = 0
                #
                # if train_accuracy>train_best_accuracy :
                #     train_best_accuracy=train_accuracy
                trigger = 0

            if trigger >= early_stop_step:
                print("=> early stopping")
                break

            # print(classification_report(val_trues, val_preds))
            # print(conf_matrix)+
            # if epoch == EPOCH - 1:
            #     plot_confusion_matrix(conf_matrix)

            val_accuracy.append(accuracy), val_presision.append(precision), val_recall.append(recall), val_f1.append(
                f1), val_loss.append(tot_val_loss.item()), val_conf_matrix.append(conf_matrix)

    result_path = 'result_' + save_name
    np.savez(result_path, val_accuracy=val_accuracy, val_presision=val_presision, val_recall=val_recall, val_f1=val_f1,
             val_loss=val_loss, val_conf_matrix=val_conf_matrix)
    after = time.time()
    total_time = after - before
    print('total_time: ' + str(total_time / 60) + ' min')
    print('best_accuracy: ' + str(best_accuracy))
    print('trigger: ' + str(trigger))


if __name__ == '__main__':
    main()
