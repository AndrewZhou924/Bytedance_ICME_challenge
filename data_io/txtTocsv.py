import numpy as np
import pandas as pd
 
txt = np.loadtxt('../final_track2_train.txt')
txtDF = pd.DataFrame(txt)
print(txtDF.shape)
txtDF.to_csv('../final_track2_train.csv',index=False)

# data = pd.read_csv("../final_track2_train.csv")
# data = data.values  # DataFrame to array
# print(data.shape) 
