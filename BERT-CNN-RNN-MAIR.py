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
LEARNING_RATE = 0.001
EPOCH = 30
SENTENCE_NUM = 50
ROUND = 10
USE_CUDA = torch.cuda.is_available()
ATTENTION_SIZE=16
USE_MAIR = False   # 是否使用Mairesse特征
MULT_HEAD = 20   #head 数目
USE_ATTENTION = 2  # 是否使用Attention层  0：表示不使用attention  1：表示使用 scaled-dot-attention   2表示使用multi-head-attention

## 保证重现
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)


def convert_to_np_arr(im_as_str):
    im = [float(i) for i in im_as_str.split()]
    im = np.asarray(im)
    return im

# 获取分类结果
def get_result(y_pred, y_test):

    macro_precision = accuracy_score(y_pred, y_test)
    print("Precision score for classification model - ", macro_precision)

    return macro_precision

# 获取每个句子的  bert 编码 以及对应的类性格的结果
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

    else:
        text_all_vec = np.load(filename + "_maxlen_25.npy")

    text_all_vec = text_all_vec[:,0:SENTENCE_NUM,:]

    return text_all_vec, cAGR,cCON,cEXT,cNEU,cOPN,mairesse_feature

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

# 模型
class CNN_Document_Encoder (nn.Module):
    def __init__(self):
        super(CNN_Document_Encoder, self).__init__()

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




    def forward(self, x):

        x = x.permute(0,2,1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x = torch.cat((x1, x2, x3, x4), dim = 1)
        x = x.view(x.size(0), -1)

        x = self.line1(x)
        x = self.line2(x)

        return x

class LSTM_Document_Encoder (nn.Module):
    def __init__(self,input_size=768, hidden_size=100,num_layers= 1, dropout=0.2, bidirectional= False, batch_first=True, attention_size  = ATTENTION_SIZE):
        super(LSTM_Document_Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.attention_size = attention_size

        if USE_CUDA:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_layers, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_layers, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers= num_layers, dropout=dropout, bidirectional= bidirectional, batch_first=batch_first)

        self.multattention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=MULT_HEAD,
                                                   dropout=self.dropout)

        self.line2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )





    def forward(self, x):

        #print(tmp.shape)


        x = x.type(torch.FloatTensor).to(device)

        batch_size = x.shape[0]
        num_directions = 0
        if self.bidirectional:
            num_directions = 2
        else:
            num_directions = 1


        h0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        #print(output.shape, hn.shape, cn.shape)

        if USE_ATTENTION == 1:
            x = self.attention_net(output)
        elif USE_ATTENTION == 2:
            query = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
            output = output.permute(1,0,2)
            attn_output, attn_output_weights = self.multattention(query, output, output)
            x = attn_output.squeeze(0)
        else:
            x = hn.squeeze(0)

        x = self.line2(x)
        return x


    def attention_net(self, lstm_output):
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.num_layers])
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, SENTENCE_NUM])
        #print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, SENTENCE_NUM, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

class CNN_LSTM_NET (nn.Module):
    def __init__(self):
        super(CNN_LSTM_NET, self).__init__()
        self.cnn_encoder = CNN_Document_Encoder()
        self.lstm_encoder = LSTM_Document_Encoder()
        self.mair_encoder = nn.Sequential(
            nn.Linear(in_features=83, out_features=50),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )

        self.line2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=2),
        )
        self.line3 = nn.Sequential(
            nn.Linear(in_features=150, out_features=2),
        )

    def forward(self, tmp):
        if USE_MAIR:

            x, x_mair = tmp
            x_mair = x_mair.type(torch.FloatTensor).to(device)
            x = x.type(torch.FloatTensor).to(device)
        else:
            x = tmp

        x1 = self.cnn_encoder(x)
        x2 = self.lstm_encoder(x)

        if USE_MAIR:
            x_mair = self.mair_encoder(x_mair)
            x = torch.cat((x1, x2, x_mair), dim = 1)
            x = self.line3(x)

        else:
            x = torch.cat((x1, x2), dim = 1)
            x = self.line2(x)

        return F.softmax(x,dim=1)


if __name__ == '__main__':

    # use GPU
    if USE_CUDA:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        torch.cuda.current_device()
        torch.cuda._initialized = True

        best_model

        text_all_vec, cAGR, cCON, cEXT, cNEU, cOPN, mairesse_feature = get_all_encode_label(encodeAgain=False)
        score_total = []
        print(EPOCH, LEARNING_RATE, BATCH_SIZE, SENTENCE_NUM)

        for cPersonality in (cAGR,):   #, cCON, cEXT, cNEU, cOPN
            score = []
            print("---------------------------------------------------------------\n")

            for i in range(ROUND):
                if USE_MAIR:
                    mairesse_feature = np.array(mairesse_feature)

                    arr_data = []

                    for i in range(len(cPersonality)):
                        arr_data.append((text_all_vec[i], convert_to_np_arr(mairesse_feature[i])))

                    # 划分测试集和训练集
                    arr_data_train, arr_data_test, y_train, y_test = train_test_split(arr_data, cPersonality,
                                                                                      test_size=TEST_RATE, random_state=0)

                else:
                    x_train, x_test, y_train, y_test = train_test_split(text_all_vec, cPersonality,
                                                                        test_size=TEST_RATE, random_state=0)

                y_test = y_test.map({"y": 1, "n": 0}).values
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
                    train_dataset = MyDataset(x_train, x_train_mair, y_train)

                else:
                    x_train = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
                    x_test = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
                    train_dataset = Data.TensorDataset(x_train, y_train)

                # 把 dataset 放入 DataLoader
                train_loader = Data.DataLoader(
                    dataset=train_dataset,  # torch TensorDataset format
                    batch_size=BATCH_SIZE,  # mini batch size
                    shuffle=True,  # 要不要打乱数据 (打乱比较好)
                    # 多线程来读数据
                )

                model = CNN_LSTM_NET().to(device)

                ##=====================================train======================================

                # optimizer 是训练的工具
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 传入 net 的所有参数, 学习率
                loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (CrossEntropyLoss)

                for epoch in range(EPOCH):  # 训练所有数据
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

                # print('Finished Training')

                # predict and evaluate ==============================================
                # Get test data loss and accuracy

                test_losses = []  # track loss
                num_correct = 0
                with torch.no_grad():
                    if USE_MAIR:
                        x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
                        x_test_mair = torch.from_numpy(x_test_mair).type(torch.FloatTensor)

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
        result_str = result_str + str(
            (EPOCH, LEARNING_RATE, BATCH_SIZE, SENTENCE_NUM, ATTENTION_SIZE, MULT_HEAD)) + "\n"
        print(EPOCH, LEARNING_RATE, BATCH_SIZE, SENTENCE_NUM)
        big_then_6 = 0
        for i in score_total:

            count = 0
            means = 0
            for j in (i):
                means += float(j)
                if (float(j) > 0.6):
                    count = count + 1

            if count > 0:
                big_then_6 = big_then_6 + 1

            print(i, count, max(i), str(means / ROUND)[0:6])
            result_str = result_str + str((i, count, max(i), str(means / ROUND)[0:6])) + '\n'

        if big_then_6 >= 2:
            print(result_str)

            with open('BERT_CNN_LSTM_result.txt', 'a') as f:
                f.write(result_str + '\n\n')

