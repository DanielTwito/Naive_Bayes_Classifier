import pandas as pd
import numpy as np


class classifier:
    def __init__(self,keys,train_data,bin_number):
        self.bin_num = bin_number
        self.keys = keys
        self.train_data =train_data
        self.numeric_attr=[]
        self.bin_data={}
        self.train_data = self.pre_process(train_data)
        self.class_option = self.extract_class()
        self.class_prob = self.calc_class_prob()
        self.m_esitmate = {}
        self.calc_cond_prob()


    def pre_process(self, train_data):
        for key in self.keys:
            if self.keys[key] == "class":
                continue
            if self.keys[key] == "NUMERIC":
                self.numeric_attr.append(key)
                train_data[key].fillna(train_data[key].mean(), inplace=True)
                self.binning(key)
            else:
                train_data[key].fillna(train_data[key].mode()[0], inplace=True)
        return train_data


    def extract_class(self):
        clss=self.keys["class"]
        clss = clss.replace(","," ").replace('{',"").replace('}',"").split()
        return clss

    def calc_class_prob(self):
        prob = {}
        total_row_number =self.train_data.shape[0]
        for clss in self.class_option:
            total_class_number = self.train_data[self.train_data["class"] == clss].shape[0]
            prob[clss] = (total_class_number*1.0) / total_row_number
        return prob


    def binning(self, key):
        min = self.train_data[key].min()
        max = self.train_data[key].max()+1
        self.bin_data[key]={'max':max,'min':min}
        bin_width = (max - min) / self.bin_num
        bins = []
        label = []
        for i in range(0, self.bin_num+1):
            label.append(str(i))
            bins.append(min + i * bin_width)
        self.keys[key]='{'+",".join(label[:len(label)-1])+'}'
        self.train_data[key] = pd.cut(x=self.train_data[key], bins=bins, labels=label[:len(label) - 1],
                                      include_lowest=True,)

    def get_bin_num(self,attr_key,value):
        max = self.bin_data[attr_key]['max']
        min = self.bin_data[attr_key]['min']
        bin_width = (max-min)/self.bin_num
        for i in range(1,self.bin_num+1):
            if min+(i-1)*bin_width <= value < min+i*bin_width:
                return i-1

    def calc_cond_prob(self):
        for key in self.keys:
            if key == "class":
                continue
            class_values = self.keys["class"].replace(",",", ").replace('{',"").replace('}',"").split(", ")
            li = self.keys[key].replace(",",", ").replace('{',"").replace('}',"").split(", ")
            for attr_val in li:
                for classify in class_values:
                    self.m_esitmate[str(key)+'='+str(attr_val)+'|'+str(classify)]= self.conditinal(key,attr_val,classify)

    def conditinal(self, key, attr_val, classify):
        # a = self.train_data[self.train_data[key] == attr_val & self.train_data["Class"] == classify].shape[0]
        filter_by_class = self.train_data[self.train_data["class"] == classify ]
        filter_by_classify_and_attrVal = filter_by_class[filter_by_class[key] == attr_val]

        # m-estimator formula
        cond_prob = (filter_by_classify_and_attrVal.shape[0]+1.0) / (filter_by_class.shape[0]+2)
        return cond_prob

    def predict(self,row):
        acc = {}
        for clss in self.class_option:
            acc[clss]=1.0
            for key in self.keys:
                value = row[key]
                if key in self.numeric_attr:
                    value = self.get_bin_num(row[key])
                prob = self.m_esitmate[key+'='+value+"|"+clss]
                acc[clss] = acc[clss] * prob
            acc[clss] = self.class_prob[clss]*acc[clss]

        pass
