import numpy as np
import matplotlib.pyplot as plt
from stochastic_gradient_descent import SGD_Linear_Regression
f
x  = np.random.rand(100,1)
y =2 + 3*x + np.random.randn(100,1)

Sgd_clf = SGD_Linear_Regression(max_iter= 1000)
Sgd_clf.fit(x,y)
y_pred = Sgd_clf.predict(x)
plt.scatter(x,y,cmap= 'Blues')
plt.plot(x,y_pred,'r-',label = 'predictions')
plt.legend()
plt.show()