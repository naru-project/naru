#%%
import numpy as np
import pandas as pd

# path = "./datasets/cover_attr_two.npy"
# column_num = 2
# columns = [6,8]
# path = "./datasets/cover_attr_0124.npy"
# column_num = 4
# columns = [0,1,2,4]
# path = "./datasets/cover_attr_six.npy"
# column_num = 6
# columns = [0,1,2,3,4,5]
path = "./datasets/cover_attr_eight.npy"
column_num = 8
columns = [0,1,2,3,4,5,6,7]


column = []
for each in columns:
    column.append(each*2)
    column.append(each*2+1)

domain = np.array([1859,3858,0,360,0,66,0,1397,-173,601,0,7117,0,254,0,254,0,254,0,7173])
raw_query = np.tile(domain,(5000,1))

df = pd.DataFrame(raw_query)
print(df.head())

range_source = np.load(path)
df[column] = range_source
print(df.head())

# %%
range_query = df.values
print(range_query[:10])
# %%
np.save("./datasets/transformed_{}.npy".format(column_num),range_query)