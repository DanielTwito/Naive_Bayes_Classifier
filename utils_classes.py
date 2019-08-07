import pandas as pd
import numpy as np

"""
This class represent a classifier.
A classifiers allow to make predictions to new incoming data based on the train set. 
"""
class classifier:
    """
    Ctor-  Defines the class members
    """
    def __init__(self,keys,train_data,bin_number):
        self.bin_num = bin_number
        self.keys = keys
        self.train_data =train_data
        self.numeric_attr={}
        self.bin_data={}
        self.train_data = self.pre_process(train_data)
        self.class_option = self.extract_class()
        self.class_prob = self.calc_class_prob()
        self.m_esitmate = {}
        self.calc_cond_prob()


    def pre_process(self, train_data):
        """
        This method responsible for data cleaning:
        Fill the missinig values where is needed and performs bining partition for NUMERIC features.
        :param train_data:
        :return:
        """
        for key in self.keys:
            if self.keys[key] == "class":
                continue
            if self.keys[key] == "NUMERIC":
                self.numeric_attr[key] = 0
                train_data[key].fillna(train_data[key].mean(), inplace=True)
                self.binning(key)
            else:
                train_data[key].fillna(train_data[key].mode()[0], inplace=True)
        return train_data

    def pre_procces_test_file(self,test_file):
        for key in self.keys:
            if key in self.numeric_attr:
                test_file[key].fillna(test_file[key].mean(), inplace=True)
            else:
                test_file[key].fillna(test_file[key].mode()[0], inplace=True)
        return test_file



    def extract_class(self):
        clss=self.keys["class"]
        clss = clss.replace(","," ").replace('{',"").replace('}',"").split()
        return clss

    def calc_class_prob(self):
        """
        Calculates the probability for each of the target class values
        :return:
        """
        prob = {}
        total_row_number =self.train_data.shape[0]
        for clss in self.class_option:
            total_class_number = self.train_data[self.train_data["class"] == clss].shape[0]
            prob[clss] = (total_class_number*1.0) / total_row_number
        return prob


    def binning(self, key):
        """
        This method responsible to divide the given numeric feature to bins.
        The partition method is "Equal Width Partitioning"
        :param key: Given column to divide
        :return: Set the new values in the given column - each value according its bin which it belongs to.
        """
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
                return str(i-1)
        if value < min:
            return '0'
        if value > max:
            return str(self.bin_num-1)

    def calc_cond_prob(self):
        """
        Calculates the conditional probability for each column
        :return:
        """
        for key in self.keys:
            if key == "class":
                continue
            class_values = self.keys["class"].replace(",",", ").replace('{',"").replace('}',"").split(", ")
            li = self.keys[key].replace(",",", ").replace('{',"").replace('}',"").split(", ")
            for attr_val in li:
                for classify in class_values:
                    self.m_esitmate[str(key)+'='+str(attr_val)+'|'+str(classify)]= self.conditinal(key,attr_val,classify)

    def conditinal(self, key, attr_val, classify):
        """
        Helper function for calculating m-estimator formula
        :return:
        """
        # a = self.train_data[self.train_data[key] == attr_val & self.train_data["Class"] == classify].shape[0]
        filter_by_class = self.train_data[self.train_data["class"] == classify ]
        filter_by_classify_and_attrVal = filter_by_class[filter_by_class[key] == attr_val]

        # m-estimator formula
        cond_prob = (filter_by_classify_and_attrVal.shape[0]+1.0) / (filter_by_class.shape[0]+2)
        return cond_prob

    def predict(self,row):
        """
        Predicts the value for the given row.
        The prediction is based on m-estimator

        :param row:
        :return:
        """
        acc = {}
        for clss in self.class_option:
            acc[clss]=1.0
            for key in self.keys:
                if key == "class":
                    continue
                value = row[key]
                if key in self.numeric_attr:
                    value = self.get_bin_num(key,row[key])
                cond_prob_in_str = key+'='+value+"|"+clss
                prob = self.m_esitmate[cond_prob_in_str]
                acc[clss] = acc[clss] * prob
            acc[clss] = self.class_prob[clss]*acc[clss]
        acc= sorted(acc.items(), key=lambda (k, v): v,reverse=True)
        return acc[0][0]
