from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import pandas as pd
import numpy as np

"""
read local data
"""
col_names = ["uid","item_id","finish_probability","like_probability"]
data = pd.read_csv("../result.csv", names=col_names)
data = data.values  # DataFrame to array
# print(data.shape)   # (2761799, 4)

finish_probability = data[:,2:3]
# print(finish_probability.shape) # (2761799, 1)
like_probability   = data[:,3:4]
# print(like_probability.shape)   # (2761799, 1)

# feature = data[:,1:2]
feature = data[:,0:2]


def getPearsonr(matrixA,matrixB):
    return (np.dot(matrixA,matrixB))/(np.sqrt(np.dot(matrixA,matrixA.T))*np.sqrt(np.dot(matrixB,matrixB.T)))


"""
calculate in different methods
"""

feature = feature[0:200,:]
finish_probability = finish_probability[0:200,:]
print(feature.shape,finish_probability.shape)

# method1:Pearsonr
# print("get pearsonr:",getPearsonr(feature,finish_probability))

# method2:计算协方差
# cov = np.cov(feature.T,finish_probability) 
# print(cov)

# method3:spearmanr
coef, pvalue = spearmanr(feature,finish_probability)
print("spearmanr-coef:",coef.shape,"\n",coef)
print("spearmanr-pvalue:",pvalue.shape,"\n",pvalue)


# method4:Kendall
# ValueError: All inputs to `kendalltau` must be of the same size, found x-size 400 and y-size 200
# coef, pvalue = kendalltau(feature,finish_probability)
# print("kendalltau-coef:",coef.shape,coef)
# print("kendalltau-pvalue:",pvalue.shape,pvalue)

