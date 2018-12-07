import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
test=pd.read_csv(r"C:\Users\onlyp\Documents\SUBLIME_TEXT_SAVES\test.csv")
test.describe()
print(test.shape)
test = test.dropna()
print(test.shape)
test.describe()
clf = LinearRegression()
X = test[['x']]
y = test['y']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)
plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions))
print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
