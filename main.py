import pandas as pd
import matplotlib.pyplot as plt
from ROC import ROC


'''
For loading data:
Make sure the data is sorted before => meaning that 

If there are multiple prob with same value => The one with positive value will come first
'''

data = pd.read_csv('./test.csv', header=0)

thresholds = data['Score']


a = ROC(data=data,pos_val='+', neg_val='-', value_col='Score', class_col='Class',plt=plt, sort_by='Score', r_col=[], thresholds="Score", label="Test", i_col='Instance'  )

a.calc_accuracy()
a.plot_curves().show()