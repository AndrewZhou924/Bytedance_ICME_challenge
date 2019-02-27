import numpy as np
import pandas as pd
import datetime


start = datetime.datetime.now()
txt = np.loadtxt('../final_track2_train.txt')
print(txt.shape)
end = datetime.datetime.now()
print("run time of reading txt",(end-start))


start = datetime.datetime.now()
csv = pd.read_csv("../final_track2_train.csv")
print(csv.shape)
end = datetime.datetime.now()
print("run time of reading csv",(end-start))