import numpy as np
import pandas as pd
import os

def getAllFiles(targetDir):
    listFiles = os.listdir(targetDir)
    return listFiles

# files = getAllFiles(r"data/small")
files = ["winequality-red.dat"]
for file in files:
    filename = file.split('.')[0]
    print(filename)
    address = file
    data_big = pd.read_table(address, delimiter=',', header=None, engine='python')

    data_big.to_csv(filename+'.csv', index=False, header=False)

# for file in files:
    # filename = file.split('.')[0]
    # print(filename)
    # address = '../data_other/' + file
    # data = pd.read_excel(address, header=None)
    # da = data.values
    # a = da[:, -1]
    # b = np.unique(a)
    # for j in range(len(b)):
    #     a = np.where(a == b[j], j+1, a)
    # da[:, -1] = a
    # data.loc[:, :] = da[:, :]
    # data.to_csv('../data_other/' + filename+'1.csv', index=False, header=False)



