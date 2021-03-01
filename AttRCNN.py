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
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import torch
import math
import torch.nn.functional as F
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 获取并且划分数据集
TRAIN_DATA_DIR = "./data/"
TEST_RATE = 0.1  # 数据集比例
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCH = 10
ROUND = 10

S = 50
W = 25
vocab_size = 0
embedding_dim = 0
GLOVE_DIR = "../2CLSTM-master/glove.6B/"


USE_CUDA = torch.cuda.is_available()

# ## 保证重现
# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)
# if USE_CUDA:
#     torch.cuda.manual_seed(1)

# use GPU
if USE_CUDA:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    torch.cuda.current_device()
    torch.cuda._initialized = True

# 获取分类结果
def get_result(y_pred, y_test):

    macro_precision = accuracy_score(y_pred, y_test)


    print("Precision score for classification model - ", macro_precision)

    return macro_precision

def Read_Text(filename = "essays.csv"):

    pd_all = pd.read_csv(os.path.join(TRAIN_DATA_DIR,filename), sep=',',header=0 )
    pd_all = pd_all

    text_all = pd_all.TEXT
    cEXT = pd_all.cEXT
    cNEU = pd_all.cNEU
    cAGR = pd_all.cAGR
    cCON = pd_all.cCON
    cOPN = pd_all.cOPN


    #计算句子和文档的平均长度
    from nltk.tokenize import sent_tokenize,word_tokenize

    sentense_num = []
    word_num = []
    for item in text_all:
        sentense_list = sent_tokenize(item)
        sentense_num.append(len(sentense_list))

        tmp = []

        for sentence in sentense_list:
            word_list = word_tokenize(sentence)
            tmp.append(len(word_list))

        word_num.append(tmp)



    sentense_num = np.array(sentense_num)
    print(sentense_num.mean())
    print(sentense_num.max())
    print(sentense_num.min())

    import pylab
    pylab.hist(sentense_num, bins=100, density=1)
    pylab.show()

    word_num = np.array(word_num)

    word_mean = []

    for item in word_num:
        item = np.array(item)
        word_mean.append(int(item.mean()))

    word_mean = np.array(word_mean)
    import pylab
    pylab.hist(word_mean, bins=500)
    pylab.show()
    print(word_mean.mean())



    return text_all, cAGR, cCON, cEXT, cNEU, cOPN


class AttRCNN (nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_weight,hidden_dim, n_layers):
        super(AttRCNN, self).__init__()

        self.embedding_dim = embedding_dim

        # sentence feature extration

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.embed.weight.data.copy_(embedding_weight)


        self.gru1 = nn.GRU(input_size=embedding_dim,hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True)
        self.mult_attention1 = nn.MultiheadAttention(embed_dim=hidden_dim ,num_heads=1, dropout=0.5)

        self.gru1_bn = nn.Sequential(nn.BatchNorm2d(num_features = hidden_dim),
                                  nn.ReLU()
        )

        self.gru2 = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                           batch_first=True)
        self.mult_attention2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, dropout=0.5)

        self.gru2_bn = nn.Sequential(nn.BatchNorm2d(num_features=hidden_dim),
                                     nn.ReLU(),
                                     )




        self.seqEncoder = nn.Sequential(
            nn.Linear(in_features= embedding_dim + 100, out_features=100),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(100, 1))
        )




        # document feature extraction

        self.conv1 = nn.Conv1d(in_channels = 100 , out_channels=50, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 100 , out_channels=100, kernel_size=1),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3,  padding=(1,))
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 100 , out_channels=100, kernel_size=3, padding=(1,)),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=5, padding=(2,))
        )


        self.conv4 = nn.Conv1d(in_channels = 100 , out_channels=50, kernel_size=3,padding=(1,))


        self.line1 = nn.Sequential(

            nn.Linear(in_features= S * 200, out_features=100),
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

        self.line3 = nn.Linear(in_features=50, out_features=2)



    def forward(self, x):
        #print(x.shape)
        x1 = x[:, :, 0 :int(x.shape[2]/2)]
        x2 = x[:, :, int(x.shape[2]/2) : x.shape[2]]

        x1 = self.embed(x1)
        x2 = self.embed(x2)

        #print(x1.shape, x2.shape)

        BATCH_SIZE_TMP =x1.shape[0]
        embedding_dim = self.embedding_dim


        x1 = x1.view(BATCH_SIZE_TMP*S, W, embedding_dim)
        x2 = x2.view(BATCH_SIZE_TMP*S, W, embedding_dim)
        x3 = x1.view(BATCH_SIZE_TMP*S, W, embedding_dim)

        #print(x1.shape, x2.shape, x3.shape)


        x1, states_1 = self.gru1(x1)
        x2, states_2 = self.gru1(x2)



        #print(x1.shape, x2.shape, x3.shape)

        x1 = x1.view(BATCH_SIZE_TMP,S, W , 50)
        x2 = x2.view(BATCH_SIZE_TMP,S, W , 50)

        x1 = self.gru1_bn(x1)
        x2 = self.gru2_bn(x2)

        x1 = x1.view(BATCH_SIZE_TMP * S, W, 50)
        x2 = x2.view(BATCH_SIZE_TMP * S, W, 50)

        x1 = x1.permute(1,0,2)
        x2 = x2.permute(1,0,2)

        #print(x1.shape)

        # query: [target length, batch size, embed dim]
        # key: [sequence length, batch size, embed dim]
        # value: [sequence length, batch size, embed dim]
        weight1 = torch.zeros((W,BATCH_SIZE_TMP * S, 50)).to(device)
        weight2 = torch.zeros((W,BATCH_SIZE_TMP * S, 50)).to(device)

        x1, attention_out = self.mult_attention1(key=x1, value=x1,query=weight1)
        x1 = x1.permute(1, 0, 2)

        x2, attention_out2 = self.mult_attention2(key=x2, value=x2, query=weight2)
        x2 = x2.permute(1, 0, 2)


        #print(x1.shape, x2.shape, x3.shape)

        x = torch.cat((x1, x3, x2), dim = 2)

        x = self.seqEncoder(x)

        x = x.squeeze(1)
        x = x.view(BATCH_SIZE_TMP,S, 100)

        #print(x.shape)

        x = x.permute(0,2,1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        #print(x1.shape, x2.shape,x3.shape,x4.shape)
        x = torch.cat((x1, x2, x3, x4), dim = 1)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.line1(x)
        x = self.line2(x)
        x = self.line3(x)

        return F.softmax(x,dim=1)

def model_visualition(model):
    # 获取预训练好的词向量和词表
    embedding_weight, word_to_idx, vocab_size = get_embedding()
    # 把单词编码成  index ，并且进行padding
    document_ints = document_pre_process(text_all, word_to_idx)
    print(model)
    x = torch.from_numpy(document_ints)
    x = x.to(device)
    y = model(x)
    g = make_dot(y)
    g.render('espnet_model2', view=False)

def pad_sentence(document_ints, dom_length, seq_length):

    if len(document_ints) < dom_length:
        for i in range(dom_length - len(document_ints)):
            document_ints.append([0]*seq_length)
    else:
        document_ints = [document_ints[i][0:seq_length] for i in range(0,dom_length)]

    return document_ints

def pad_words(sentence_ints, seq_length, reverse = False):

    if (reverse):
        sentence_ints.reverse()

    if len(sentence_ints) < seq_length:

        sentence_ints = list(sentence_ints + [0] * (seq_length -  len(sentence_ints)))
    else :
        sentence_ints = sentence_ints[0:seq_length]

    return sentence_ints

def document_pre_process (text_all, word_to_idx):
    total_ints = []
    # 对于每个文档
    for document in text_all:

        document_ints = []  # document_ints  文档中的index表示

        # 分句
        sentence_list = sent_tokenize(document)
        # 对于每个句子
        for sentence in sentence_list:
            # 去标点和分词

            sentence_ints = []  # sentence_ints  文档中的每一句的index表示

            tokenizer = RegexpTokenizer(r'\w+')
            sentence = tokenizer.tokenize(sentence)
            # 对于每个词， 找到对应的index
            for word in sentence:
                try:
                    sentence_ints.append(word_to_idx[word])
                except:
                    sentence_ints.append(word_to_idx['<unk>'])

            # 对句子进行填充至每句25个word
            sentence_ints_no_reverse = pad_words(sentence_ints, W, reverse=False)
            sentence_ints_reverse = pad_words(sentence_ints, W, reverse=True)

            sentence_ints = sentence_ints_no_reverse + sentence_ints_reverse

            document_ints.append(sentence_ints)

        # 对文档进行填充至每个文档50个sentence
        document_ints = pad_sentence(document_ints, S, W * 2)

        total_ints.append(document_ints)

    total_ints = np.array(total_ints)
    print(total_ints.shape)
    return total_ints

def get_embedding(text_all):
    all_text = ""

    for item in text_all:

        tokenizer = RegexpTokenizer(r'\w+')
        sentences = tokenizer.tokenize(item)
        sentences_str = ' '.join(sentences)
        all_text += sentences_str


    # create a list of words
    words = word_tokenize(all_text)
    vocab = set(words)
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    # 定义了一个unknown的词，也就是说没有出现在训练集里的词，我们都叫做unknown，词向量就定义为0。
    word_to_idx['<unk>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'

    '''转换向量过程'''
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors


    #wvmodel = KeyedVectors.load_word2vec_format("test_word2vec.txt")
    wvmodel = KeyedVectors.load_word2vec_format("glove.6B.100d_word2vec.txt")
    vocab_size = len(vocab) + 1
    embed_size = 100

    print(vocab_size, embed_size)

    embedding_weight = torch.zeros(vocab_size, embed_size)

    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except:
            continue
        embedding_weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

    return embedding_weight, word_to_idx, vocab_size, embed_size

if __name__ == '__main__':

     text_all, cAGR, cCON, cEXT, cNEU, cOPN = Read_Text()
     score_total = []

     for cPersonality in (cAGR, cCON, cEXT, cNEU, cOPN):

         text_all = text_all
         cAGR = cAGR

         # 获取预训练好的词向量和词表
         embedding_weight , word_to_idx, vocab_size, embed_size  =  get_embedding(text_all)
         # 把单词编码成  index ，并且进行padding
         document_ints = document_pre_process(text_all, word_to_idx)

         # 划分测试集和训练集
         x_train, x_test, y_train, y_test = train_test_split(document_ints, cAGR, test_size=TEST_RATE,random_state=0)

         y_test = y_test.map({"y": 1, "n": 0}).values
         pd.to_numeric(y_test)

         y_train = y_train.map({"y": 1, "n": 0}).values
         pd.to_numeric(y_train)
         y_train = torch.from_numpy(to_categorical(y_train))

         print('train size        ' + str(x_train.shape))
         print('test  size        ' + str(x_test.shape))

         x_train = torch.from_numpy(x_train)
         x_test = torch.from_numpy(x_test)

         train_dataset = Data.TensorDataset(x_train, y_train)

         # 把 dataset 放入 DataLoader
         train_loader = Data.DataLoader(
             dataset=train_dataset,  # torch TensorDataset format
             batch_size=BATCH_SIZE,  # mini batch size
             shuffle=True,  # 要不要打乱数据 (打乱比较好)
             # 多线程来读数据
         )
         score = []

         for i in range(ROUND):
             model = AttRCNN(vocab_size=vocab_size, hidden_dim=50, n_layers=1, embedding_dim=embed_size, embedding_weight=embedding_weight,
                             ).to(device)



             ##=====================================train======================================

             # optimizer 是训练的工具
             optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 传入 net 的所有参数, 学习率
             loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (CrossEntropyLoss)

             for epoch in range(EPOCH):  # 训练所有!整套!数据 3 次
                 running_loss = 0.0

                 for step, (batch_x, batch_y) in enumerate(train_loader):  # 每一步 loader 释放一小批数据用来学习

                     batch_x = batch_x.to(device)
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

             print('Finished Training')

             # predict and evaluate ==============================================
             # Get test data loss and accuracy

             test_losses = []  # track loss
             num_correct = 0
             with torch.no_grad():
                 x_test = x_test.to(device)

                 y_proba = model(x_test)
                 y_pred = y_proba.cpu().numpy()
                 y_pred = np.argmax(y_pred, axis=1)
                 result = get_result(y_pred, y_test)

                 print(result)
             score.append(str(result)[0:6])

         score_total.append(score)

     for i in score_total:

         count = 0
         for j in (i):
             if (float(j) > 0.6):
                 count = count + 1

         print(i, count, max(i))
