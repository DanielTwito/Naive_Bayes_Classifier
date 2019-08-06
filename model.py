import csv
import os
from utils_classes import classifier


class model:

    def __init__(self):
        self.train_model=None
        self.navie_classifier = None

        pass

    def create_classifier(self, keys, train,bin_num):
        features = self.extract_features(keys)
        self.navie_classifier = classifier(features, train,bin_num)
        pass

    def extract_features(self, keys):
        feature = {}
        for key in keys:
            tmp = key.split()
            feature[tmp[1]]=" ".join(tmp[2:])
        return feature