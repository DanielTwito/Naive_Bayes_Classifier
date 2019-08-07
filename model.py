import csv
import os
from utils_classes import classifier


class model:

    def __init__(self):
        """
        Ctor
        """
        self.train_model = None
        self.navie_bayes_classifier = None
        self.expected_files = ["Structure.txt", "test.csv", "train.csv"]

    def validate_files(self, file_list):
        """
        This method validates that the nesseary files are exsits
        :param file_list: missing files
        :return:
        """
        ans = []
        for file in self.expected_files:
            if file in file_list:
                continue
            else:
                ans.append(file)
        return ans

    def create_classifier(self, keys, train, bin_num):
        """
        This method creates a classifier with the given parameters
        :param keys: data features
        :param train: dataset of the train data
        :param bin_num:
        :return:
        """
        features = self.extract_features(keys)
        self.navie_bayes_classifier = classifier(features, train, bin_num)

        pass

    def extract_features(self, keys):
        """
        This method extracts the features from the given input.
        :return list of the features with their possible values
        """
        feature = {}
        for key in keys:
            tmp = key.split()
            feature[tmp[1]] = " ".join(tmp[2:])
        return feature

    def execute_classification(self, test, output_path):
        """
        This methid responsible for running classification
        :param test: given dataset of the test data
        :param output_path: A path to save the output file
        :return: save classification results in "output.txt" file
        """
        test = self.navie_bayes_classifier.pre_procces_test_file(test)
        with open(output_path, "w") as output:
            for index, row in test.iterrows():
                prediction = self.navie_bayes_classifier.predict(row)
                output.write(str(index + 1) + " " + str(prediction) + "\n")
