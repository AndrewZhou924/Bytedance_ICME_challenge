from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

"""
input local data and proccess
"""
# col_names = ["uid","item_id","finish_probability","like_probability"]
# data = pd.read_csv("../result.csv", names=col_names)
# data = data.values  # DataFrame to array
# print(data.shape)   # (2761799, 4)

# finish_probability = data[:,2:3]
# print(finish_probability.shape) # (2761799, 1)
# like_probability   = data[:,3:4]
# print(like_probability.shape)   # (2761799, 1)

# feature = data[:,1:2]
# feature = data[:,0:2]
# print(feature.shape)

# read data from final_track2_train.csv
data2_col_names = ["uid","user_city","item_id","author_id","item_city","channel","finish","like","music_id","did","creat_time","video_duration"]
data2 = pd.read_csv("../final_track2_train.csv", names=data2_col_names)
data2 = data2.values
# data2 = data2[0:100,:]
print(data2.shape)
finish = data2[:,6:7]
like   = data2[:,7:8]

"""
select feature
"""
# 方法 1 : 相关系数法
#####################################################################################################################
# return of pearsonr(): (Pearson’s correlation coefficient,2-tailed p-value)

# print("np.corrcoef: ",np.corrcoef(feature, finish_probability))
# print("finish_probability",pearsonr(feature.all(), finish_probability))
# SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T,k=2).fit_transform(feature, finish_probability)
# SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T,k=2).fit_transform(feature, like_probability)

# get pearsonr for feature in final_track2_train.csv
for i in range(len(data2_col_names)):
    feature = data2[:,i:(i+1)]
    score1 = pearsonr(feature, finish)
    score2 = pearsonr(feature, like)
    print(data2_col_names[i]," ---score on finish:", score1, " ---score on like:",score2)


# 方法 2 : 
#####################################################################################################################
