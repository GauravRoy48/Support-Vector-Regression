##################################################################################
# Creator     : Gaurav Roy
# Date        : 12 May 2019
# Description : The code contains the approach for Polynomial Linear Regression on 
#               the Position_Salaries.csv.
##################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma = 'auto')
regressor.fit(X,Y)

# Predicting a new result with SVR
Y_pred =sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing SVR results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid()
plt.show()

