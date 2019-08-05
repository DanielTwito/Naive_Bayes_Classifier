import pandas as pd
import numpy as np


class classifier:
    def __init__(self,keys,train_data):
        self.keys = keys
        a = train_data
        self.train_data = self.pre_process(train_data)
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




