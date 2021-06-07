#%%
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# path = "./results/results_2.csv"
# MAX_CARD = 501012

# path = './results/results_dmv_all.csv'
# MAX_CARD = 9406943

path = './results/tpch_result.csv'

df = pd.read_csv(path)
MAX_CARD = 6000003 * 0.2
# print(df)


est_card = df['est_card'].values/MAX_CARD
true_card = df['true_card'].values/MAX_CARD


print("RMSE error:",np.sqrt(mean_squared_error(est_card,true_card)))


# %%
def print_qerror(pred, label):
    qerror = []
    for i in range(len(pred)):
        if pred[i]==0 and float(label[i])==0:
            qerror.append(1)
        elif pred[i]==0:
            qerror.append(label[i])
        elif label[i]==0:
            qerror.append(pred[i])
        elif pred[i] > float(label[i]):
            qerror.append(float(pred[i]) / float(label[i]))
        else:
            qerror.append(float(label[i]) / float(pred[i]))
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

print_qerror(est_card,true_card)
# %%
