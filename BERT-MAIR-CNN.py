# -*- coding: utf-8 -*-
# @File  : pyt_train.py
# @Author: wangjiwu
# @Date  : 2020/2/22
# @Desc  :

import os
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.utils.data as Data
import torch
from torchviz import make_dot
from torch.autograd import Variable

import torch.nn.functional as F
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 获取并且划分数据集
TRAIN_DATA_DIR = "./data/"
TEST_RATE = 0.1
BATCH_SIZE = 512
LEARNING_RATE = 0.01
EPOCH = 20
ROUND = 10
USE_MAIR = True  # 是否加入Mairesse进行训练
SENTENCE_NUM = 50
USE_CUDA = torch.cuda.is_available()

## 保证重现
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

# 获取分类结果
def get_result(y_pred, y_test):

    macro_precision = accuracy_score(y_pred, y_test)


    print("Precision score for classification model - ", macro_precision)

    return macro_precision

# 获取每个句子的  bert 编码 以及对应的类性格的结果
# encoderAgain 表示是否重新对句子进行BERT编码， False表示不编码而读取以前的句子编码

def get_all_encode_label(filename = "essays_mairesse.csv", encodeAgain = False):

    pd_all = pd.read_csv(os.path.join(TRAIN_DATA_DIR,filename), sep=',',header=0 )
    pd_all = pd_all
    mairesse_feature = pd_all.mairesse_feature
    text_all = pd_all.TEXT
    cEXT = pd_all.cEXT
    cNEU = pd_all.cNEU
    cAGR = pd_all.cAGR
    cCON = pd_all.cCON
    cOPN = pd_all.cOPN

    if encodeAgain == True:


        from nltk.tokenize import sent_tokenize
        from bert_serving.client import BertClient

        #bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4

        bc = BertClient()
        text_all_vec = []

        import time

        time_start = time.time()

        for item in text_all:
            sentense_list = sent_tokenize(item)
            text_sentence_vector = bc.encode(sentense_list)



            padding_num = 336 - len(sentense_list)
            padding_vec = np.zeros((padding_num, 768), dtype=float)
            text_sentence_vector = np.concatenate((text_sentence_vector, padding_vec), axis=0)
            text_all_vec.append(text_sentence_vector)

        time_end = time.time()
        print('BERT encoding time cost: ', time_end - time_start, 's')
        np.save( filename +"_maxlen_25.npy",text_all_vec)


    text_all_vec = np.load(filename + "_maxlen_25.npy")



    text_all_vec = text_all_vec[:,0:SENTENCE_NUM,:]

    return text_all_vec, cAGR,cCON,cEXT,cNEU,cOPN, mairesse_feature

def convert_to_np_arr(im_as_str):
    im = [float(i) for i in im_as_str.split()]
    im = np.asarray(im)
    return im

# 模型
class Bert_Net (nn.Module):
    def __init__(self):
        super(Bert_Net, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 768 , out_channels=50, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 768 , out_channels=100, kernel_size=1),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3,  padding=(1,))
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 768 , out_channels=100, kernel_size=3, padding=(1,)),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=5, padding=(2,))
        )

        self.conv4 = nn.Conv1d(in_channels = 768 , out_channels=50, kernel_size=3,padding=(1,))


        self.line1 = nn.Sequential(
            nn.Linear(in_features= SENTENCE_NUM* 200, out_features=100),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU()
        )

        self.line2 = nn.Sequential(

            nn.Linear(in_features= 100, out_features=50),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )
        self.line4 = nn.Sequential(
            nn.Linear(in_features=83, out_features=50),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )

        if USE_MAIR:
            self.line3 = nn.Linear(in_features=100, out_features=2)
        else:
            self.line3 = nn.Linear(in_features=50, out_features=2)





    def forward(self, tmp):

        if USE_MAIR:
            x, x_mair = tmp
            x_mair = x_mair.type(torch.FloatTensor).to(device)
            x = x.permute(0,2,1).type(torch.FloatTensor).to(device)
        else:
            x = tmp.permute(0,2,1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x = torch.cat((x1, x2, x3, x4), dim = 1)
        x = x.view(x.size(0), -1)

        x = self.line1(x)
        x = self.line2(x)
        if USE_MAIR:
            x_mair = self.line4(x_mair)
            x = torch.cat((x, x_mair), dim = 1)

        x = self.line3(x)


        return F.softmax(x,dim=1)



from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self, x, x_mair, y ):
        self.x = x
        self.x_mair = x_mair
        self.y = y

    def __getitem__(self, index):
        input = self.x[index], self.x_mair[index]
        output = self.y[index]
        return input, output

    def __len__(self):
        return len(self.x_mair)



if __name__ == '__main__':


    # use GPU
    if USE_CUDA:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        torch.cuda.current_device()
        torch.cuda._initialized = True
    for LEARNING_RATE in (0.05,0.005):
        for EPOCH in (10, 20,30):
            for SENTENCE_NUM in (50, 100 ,200, 336):
                text_all_vec, cAGR, cCON, cEXT, cNEU, cOPN, mairesse_feature = get_all_encode_label(encodeAgain=False)
                score_total = []
                print(EPOCH, LEARNING_RATE, BATCH_SIZE, SENTENCE_NUM)

                for cPersonality in (cAGR,cCON, cEXT, cNEU, cOPN ):
                    score = []
                    print("---------------------------------------------------------------\n")

                    for i in range(ROUND):
                        if USE_MAIR:
                            mairesse_feature = np.array(mairesse_feature)

                            arr_data = []

                            for i in range(len(cPersonality)):
                                arr_data.append((text_all_vec[i],convert_to_np_arr(mairesse_feature[i])))


                            # 划分测试集和训练集
                            arr_data_train, arr_data_test, y_train, y_test = train_test_split(arr_data, cPersonality, test_size=TEST_RATE,random_state=0)

                        else :
                            x_train, x_test, y_train, y_test = train_test_split(text_all_vec, cPersonality,
                                                                                              test_size=TEST_RATE, random_state=0)

                        y_test =  y_test.map({"y":1, "n": 0}).values
                        pd.to_numeric(y_test)

                        y_train = y_train.map({"y": 1, "n": 0}).values
                        pd.to_numeric(y_train)
                        y_train = torch.from_numpy(to_categorical(y_train))

                        if USE_MAIR:
                            x_train = []
                            x_train_mair = []
                            x_test = []
                            x_test_mair = []

                            for item in arr_data_train:
                                x_train.append(item[0])
                                x_train_mair.append(item[1])
                            for item in arr_data_test:
                                x_test.append(item[0])
                                x_test_mair.append(item[1])

                            x_train = np.asarray(x_train)
                            x_test = np.asarray(x_test)
                            x_train_mair = np.asarray(x_train_mair)
                            x_test_mair = np.asarray(x_test_mair)

                        if USE_MAIR:
                            train_dataset = MyDataset(x_train, x_train_mair, y_train)
                        else :
                            x_train = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
                            x_test = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
                            train_dataset = Data.TensorDataset(x_train, y_train)

                        # 把 dataset 放入 DataLoader
                        train_loader = Data.DataLoader(
                            dataset=train_dataset,      # torch TensorDataset format
                            batch_size=BATCH_SIZE,      # mini batch size
                            shuffle=True,               # 要不要打乱数据 (打乱比较好)
                        )



                        model = Bert_Net().to(device)

                        ##=====================================train======================================

                        # optimizer 是训练的工具
                        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 传入 net 的所有参数, 学习率
                        loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (CrossEntropyLoss)

                        for epoch in range(EPOCH):  # 训练所有!整套!数据 3 次
                            running_loss = 0.0

                            for step, (batch_x, batch_y) in enumerate(train_loader):  # 每一步 loader 释放一小批数据用来学习
                                batch_y = batch_y.to(device)
                                optimizer.zero_grad()

                                out = model(batch_x)  # 喂给 net 训练数据 x, 输出分析值

                                # print(out)
                                loss = loss_func(out, batch_y)  # 计算两者的误差
                                loss.backward()  # 误差反向传播, 计算参数更新值
                                optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

                                running_loss += loss.data

                                if step % BATCH_SIZE == 0:
                                    print('[epoch %d] loss: %.4f' %
                                          (epoch + 1, running_loss))
                                    running_loss = 0.0

                        #print('Finished Training')

                        # predict and evaluate ==============================================
                        # Get test data loss and accuracy

                        test_losses = []  # track loss
                        num_correct = 0
                        with torch.no_grad():
                            if USE_MAIR:
                                x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
                                x_test_mair =  torch.from_numpy(x_test_mair).type(torch.FloatTensor)

                                x_test_tuple = (x_test, x_test_mair)
                            else:
                                x_test_tuple = x_test
                            y_proba = model(x_test_tuple)
                            y_pred = y_proba.cpu().numpy()
                            y_pred = np.argmax(y_pred, axis=1)
                            result = get_result(y_pred, y_test)

                        score.append(str(result)[0:6])



                    score_total.append(score)

                result_str = ' '
                result_str = result_str + str((EPOCH, LEARNING_RATE, BATCH_SIZE, SENTENCE_NUM)) + "\n"
                print(EPOCH, LEARNING_RATE, BATCH_SIZE, SENTENCE_NUM)
                big_then_6 = 0
                for i in score_total:

                    count = 0
                    means = 0
                    for j in (i):
                        means += float(j)
                        if (float(j) > 0.6):
                            count = count + 1

                    if count >0 :
                        big_then_6 = big_then_6 + 1

                    print(i, count, max(i), str(means/ROUND)[0:6])
                    result_str = result_str +  str((i, count, max(i), str(means/ROUND)[0:6])) + '\n'


                if big_then_6 >= 2:
                    print(result_str)

                    with open('BERT_CNN_Mairesse_result.txt', 'a') as f:
                        f.write(result_str + '\n\n')

