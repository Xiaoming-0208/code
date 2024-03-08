from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
plt.rc('font', family='Times New Roman')

data = pd.read_csv('kriging_result.csv')


x = pd.DataFrame(data, columns=['C', 'F'], dtype=float)
y = pd.DataFrame(data, columns=['S', 'P', 'K'], dtype=float)
mean_x = np.mean(x)
std_x = np.std(x)
x = (x-mean_x)/std_x

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)

model_mlp = MLPRegressor(
    hidden_layer_sizes=(128, 128, 128),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=128,
    learning_rate='constant',
    learning_rate_init=0.002,
    power_t=0.5,
    max_iter=2000,
    shuffle=True,
    tol=0.001,
    verbose=True,
    warm_start=True,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-10,
    random_state=9,
)
model_mlp.n_outputs_ = 3
model_mlp.fit(X_train, Y_train)
joblib.dump(model_mlp, 'model_mlp_KSANN.pkl')

mlp_score = model_mlp.score(X_train, Y_train)

pred_train = model_mlp.predict(X_train)

from sklearn.metrics import mean_squared_error
mse_1 = mean_squared_error(Y_train, pred_train)

pred_test = model_mlp.predict(X_test)
mse_2 = mean_squared_error(Y_test, pred_test)

test_data = pd.read_csv('Traditional_ANN_test_input_data.csv')

test_x = pd.DataFrame(test_data, columns=['C', 'F'], dtype=float)
test_x = (test_x-mean_x)/std_x
test_y = pd.DataFrame(test_data, columns=['S', 'P', 'K'], dtype=float)

test_predic = model_mlp.predict(test_x)

plt.figure(figsize=(7, 6))
t1 = plt.scatter(test_y.iloc[:, 0], test_predic[:, 0], alpha=0.6, c='brown', linewidths=1.0, edgecolors='black', s=100)
t2 = plt.scatter(test_y.iloc[:, 1], test_predic[:, 1], alpha=0.6, c='darkviolet', linewidths=1.0, edgecolors='black', s=100)
t3 = plt.scatter(test_y.iloc[:, 2], test_predic[:, 2], alpha=0.6, c='darkorange', linewidths=1.0, edgecolors='black', s=100)
plt.xlim(0.0, 40)
plt.ylim(0.0, 40)
my_x_ticks = np.arange(0.0, 40.001, 5)
my_y_ticks = np.arange(0.0, 40.001, 5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
XX = np.linspace(0.0, 38, 10)
YY = XX
plt.plot(XX, YY, linewidth=1.5, color='k')
plt.tick_params(labelsize=15)

plt.xlabel("Actual value", size=18)
plt.ylabel("Predicted value", size=18)
plt.legend((t1, t2, t3), (u'Strength', u'porosity', u'Permeability'), loc=2, prop={'size': 16})
#plt.savefig('result.jpg', dpi=800)
plt.show()

R2_test = r2_score(test_y, test_predic)

print(mlp_score)
print(mse_1)
print(mse_2)
print(R2_test)
