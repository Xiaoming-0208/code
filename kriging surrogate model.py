import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import random

#plt.rc('font', family='Times New Roman')

Data5_100 = pd.read_csv('P_5_100.csv')
Data5_150 = pd.read_csv('P_5_150.csv')
Data5_200 = pd.read_csv('P_5_200.csv')
Data10_100 = pd.read_csv('P_10_100.csv')
Data10_150 = pd.read_csv('P_10_150.csv')
Data10_200 = pd.read_csv('P_10_200.csv')
Data20_100 = pd.read_csv('P_20_100.csv')
Data20_150 = pd.read_csv('P_20_150.csv')
Data20_200 = pd.read_csv('P_20_200.csv')

grid_x = np.arange(5, 21.0, 0.5)
grid_y = np.arange(10, 21.0, 0.5)

n = 1
sim_data = []
for i in range(0, 1499):
    m_1 = (Data5_100.iloc[n,:],
           Data5_150.iloc[n,:],
           Data5_200.iloc[n,:],
           Data10_100.iloc[n,:],
           Data10_150.iloc[n,:],
           Data10_200.iloc[n,:],
           Data20_100.iloc[n,:],
           Data20_150.iloc[n,:],
           Data20_200.iloc[n,:],
           )
    m_2 = pd.DataFrame(m_1)
    m_3 = np.array(m_2)

    OK = OrdinaryKriging(
        m_3[:,0],
        m_3[:,1],
        m_3[:,2],
        variogram_model='linear',
        weight=True,
        exact_values=True,
        pseudo_inv=False,
        enable_plotting=False,
    )

    z, ss = OK.execute('grid', grid_x, grid_y)

    X = np.asarray(grid_x, 'float32')
    Y = np.asarray(grid_y, 'float32')
    len_X = len(X)
    len_Y = len(Y)

    plt.imshow(z, origin='lower', cmap='viridis')
    plt.show()

    out = []
    for j in range(0, len_X-1):
        for k in range(0, len_Y-1):
            out_data = (X[j], 10*Y[k], z[k, j])
            out.append(out_data)

    sim_data.append(out)
    number = random.uniform(5, 30)
    n += int(number)

df = pd.DataFrame(sim_data)
df2 = pd.concat(df.iloc[:, i] for i in range(df.shape[1]))
df3 = [', '.join(map(str, x)) for x in df2]
df4 = pd.DataFrame(df3)
df4 = df4[0].str.split(',', expand=True)
sim_result = pd.DataFrame(df4)

#sim_result.to_csv('kriging_P_result.csv')
#print(sim_result)



