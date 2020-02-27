import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import LinearRegression
import pandas as pd

plt.style.use('ggplot')
x = (pd.read_csv('guassian_distribution.csv',header = None)[0]).to_numpy().reshape(-1,1)
y = (pd.read_csv('guassian_distribution.csv',header = None)[1]).to_numpy().reshape(-1,1)
std = np.std(x)  # np.std returns ndarray
mean = np.mean(x)
x = (x-mean)/std
x_plot= np.sort(x,axis=0 )
lr = LinearRegression()
lr.fit(x,y)
print(lr.base_i,lr.base_t)
y_pred = lr.predict(x_plot)
plt.scatter(x,y)
plt.plot(x_plot,y_pred,'g--')

plt.show()
