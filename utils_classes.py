import pandas as pd
import numpy as np


class classifier:
    def __init__(self,keys,train_data):
        self.keys = keys
        a = train_data
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


