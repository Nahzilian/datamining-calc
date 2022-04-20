from matplotlib import pyplot
from pandas import DataFrame
from helpers.preprocess import preprocess
from helpers.confusion_matrix import cm_converter, cm_converter_custom_threshold
# from helpers.merge_sort import merge_sort
# import matplotlib.pyplot as plt

class ROC:
    def __init__(self, pos_val, neg_val, value_col, class_col, data: DataFrame, plt: pyplot, sort_by: list[str] = [], r_col: list[str] = [], thresholds: str = '', label: str = '', nan_val = None, i_col:str= '') -> None:
        self.data = preprocess(data, sort_by, nan_val, r_columns=r_col)
        self.plt = plt
        self.tpr = []
        self.fpr = []
        self.thresholds = thresholds

        self.pos_val = pos_val
        self.neg_val = neg_val
        self.value_col = value_col
        self.class_col = class_col
        self.i_col = i_col

        self.label = label

    def calc_accuracy(self):
        threshold_col = self.data[self.thresholds].tolist()

        for index, threshold in enumerate(threshold_col):
            cm = cm_converter(self.data, threshold, self.pos_val,self.neg_val, self.value_col, self.class_col, index + 1, self.i_col)

            self.tpr.append(cm.tpr)
            self.fpr.append(cm.fpr)
        # print(self.tpr)
        # print(self.fpr)

    def calc_measurements_with_threshold(self, threshold):
        # Where to get the index????
        cm = cm_converter_custom_threshold(self.data, threshold, self.pos_val,self.neg_val, self.value_col, self.class_col)
        return cm.get_measurements()


    def plot_curves(self):
        
        self.plt.plot(self.fpr, self.tpr, label=self.label)
        self.plt.scatter(self.fpr, self.tpr)

        self.plt.legend()
        return self.plt
