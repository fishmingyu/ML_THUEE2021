import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import itertools
import re

class DataLoader(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.process()

    def process(self):
        df = pd.read_csv(self.file_path, sep=',', encoding='latin')
        df.reset_index(drop=True)
        name_dict = dict()
        new_name = ['label', 'sms', 'aux1', 'aux2', 'aux3'] # 标定数据集
        for i in range(len(df.columns)):
            name_dict[df.columns[i]] = new_name[i] #重命名
        self.df = df.rename(columns=name_dict)
        self.label = self.df['label']
        self.sms = self.df['sms'] # 目前只用sms的信息
        self.message_num = len(self.df)
        data_processed, label_processed = self.genData()
        self.train_test_split(np.array(data_processed), np.array(label_processed))
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) # not character, abolished 
        text_list = text.split()
        return text_list

    def train_test_split(self, X, y):
        # 由于初始数据集，分成采样
        ss = StratifiedShuffleSplit(n_splits=1,test_size=0.35,random_state=42)# test_frac 0.35
        for train_index, test_index in ss.split(X, y):
            print("TRAIN number:", len(train_index), "TEST number", len(test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        self.train_data = [X_train, y_train]
        self.test_data = [X_test, y_test]
        
    def genData(self):
        sms_dict_list = []
        sms_data_list = []
        max_len = 0
        # count data
        for message in self.sms:
            text_list = self.clean_text(message) # lower and regularize
            for i in text_list:
                if i != "":
                    sms_dict_list.append(i) # add to dict
            text_list.append('EOS') # add end of sentence
            if(len(text_list) > max_len):
                max_len = len(text_list)
            sms_data_list.append(text_list)
        self.senten_len = max_len

        # gen dictionary
        sms_unique = np.unique(sms_dict_list).tolist() # statistics
        sms_unique.append("EOS")
        sms_unique.append("PAD")
        idx = list(range(0, len(sms_unique)))
        self.charc_dict = dict(zip(sms_unique, idx))
        self.word_num = len(self.charc_dict)

        # padding and reset
        sms_processed = []
        for m in range(self.message_num):
            frag = sms_data_list[m]
            new_frag = []
            for i in range(max_len):
                if i < len(frag):
                    tmp = frag[i]
                    key = self.charc_dict[tmp] # search key
                    new_frag.append(key)
                else:
                    new_frag.append(self.charc_dict["PAD"]) # add padding
            sms_processed.append(new_frag)
        self.sms_data = sms_processed

        # label to number
        y = []
        for i in self.label:
            if i == 'ham':
                y.append(0)
            elif i == 'spam':
                y.append(1)
        return sms_processed, y
