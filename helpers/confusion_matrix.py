from pandas import DataFrame


class ConfusionMatrix:
    def __init__(self, tp, fp, tn, fn) -> None:
        self.tp = tp
        self.fn = fn
        self.tn = tn
        self.fp = fp
        self.tpr = self.tp / (self.tp + self.fn)
        self.fpr = self.fp / (self.fp + self.tn)
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recal(self):
        return self.tp / (self.tp + self.fn)
    
    def get_f_measure(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

def cm_converter (data: DataFrame, threshold, pos_val, neg_val, col, class_col, index, index_col):
    '''
    tp: all above value that match positive with threshold
    fp: all above expected to match positive but appear negative in dataframe
    tn: all below value that match negative
    fn: all below expected match negative but appear positive in dataframe
    '''

    positive_rows = data[col] >= threshold
    negative_rows = data[col] < threshold
    
    
    # assuming that data has been sorted and index is also correct
    is_tp = data[(data[col] >= threshold) & (data[class_col] == pos_val) & (data[index_col] <= index)]
    is_tn = data[(data[col] <= threshold) & (data[class_col] == neg_val) & (data[index_col] > index)]
    is_fp = data[(data[col] >= threshold) & (data[class_col] == neg_val) & (data[index_col] <= index)]
    is_fn = data[(data[col] <= threshold) & (data[class_col] == pos_val)  & (data[index_col] > index)]

    # tp, fp, tn, fn = 3 3 2 2
    # tp, fp, tn, fn = 4 4 1 1 --- 0.85 -- id 6

    tp, fp, tn, fn = is_tp.shape[0], is_fp.shape[0], is_tn.shape[0], is_fn.shape[0]
    # print(tp, fp, tn, fn)
    return ConfusionMatrix(tp, fp, tn, fn)

