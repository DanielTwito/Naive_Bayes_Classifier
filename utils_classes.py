import pandas as pd
import numpy as np


class classifier:
    def __init__(self,keys,train_data,bin_number):
        self.bin_num = bin_number
        self.keys = keys
        self.train_data =train_data
        self.train_data = self.pre_process(train_data)
        self.class_option = self.extract_calss()
        self.class_prob = self.calc_class_prob()
        self.m_esitmate = {}

    def pre_process(self, train_data):
        for key in self.keys:
            if self.keys[key] == "Class":
                continue
            if self.keys[key] == "NUMERIC":
                train_data[key].fillna(train_data[key].mean(), inplace=True)
                self.binning(key)
            else:
                train_data[key].fillna(train_data[key].mode()[0], inplace=True)
        return train_data

    def extract_calss(self):
        clss=self.keys["class"]
        clss = clss.replace(","," ").replace('{',"").replace('}',"").split()
        return clss

    def calc_class_prob(self):
        prob = {}
        total_row_number =self.train_data.shape[0]
        for clss in self.class_option:
            total_class_number = self.train_data[self.train_data["class"] == 'Y'].shape[0]
            prob[clss] = (total_class_number*1.0) / total_row_number
        print prob


    def binning(self, key):
        min =self.train_data[key].min()-1
        max =self.train_data[key].max()+1
        bin_width= (max-min)/self.bin_num
        bins = []
        label =[]
        for i in range(0,self.bin_num+1):
            label.append(str(i))
            bins.append(min + i*bin_width)
        self.train_data[key] = pd.cut(x=self.train_data[key], bins=bins,labels=label[:len(label)-1])


