import csv
import os
from utils_classes import classifier


class model:

    def __init__(self):
        self.train_model = None
        self.navie_bayes_classifier = None
        self.expected_files = ["Structure.txt", "test.csv", "train.csv"]

    def validate_files(self, file_list):
        ans = []
        for file in self.expected_files:
            if file in file_list:
                continue
            else:
                ans.append(file)
        return ans

    def create_classifier(self, keys, train, bin_num):
        features = self.extract_features(keys)
        self.navie_bayes_classifier = classifier(features, train, bin_num)

        pass

    def extract_features(self, keys):
        feature = {}
        for key in keys:
            tmp = key.split()
            feature[tmp[1]] = " ".join(tmp[2:])
        return feature

    def execute_classification(self, test, output_path):
        test = self.navie_bayes_classifier.pre_procces_test_file(test)
        with open(output_path, "w") as output:
            for index, row in test.iterrows():
                prediction = self.navie_bayes_classifier.predict(row)
                output.write(str(index + 1) + " " + str(prediction) + "\n")
