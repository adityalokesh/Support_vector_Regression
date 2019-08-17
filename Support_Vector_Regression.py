#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:26:06 2019

@author: chrx
"""
# creating a SUPPORT VECTOR REGRESSION MODEL

#importing nessecary packages and data 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('/home/chrx/Downloads/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')

#splitting data as dependent and independent variable
X=data.iloc[:, 1:2].values
Y=data.iloc[:, 2:3].values


#feature scaling the data as SVR model does not automatically support 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
sc_y=StandardScaler()
Y=sc_y.fit_transform(Y)

# creating a SVR model with gaussian kernel
from sklearn.svm import SVR
svr_obj=SVR(kernel='rbf')
svr_obj.fit(X,Y)

svr_poly=SVR(kernel='poly')
svr_poly.fit(X,Y)

svr_sig=SVR(kernel='sigmoid')
svr_sig.fit(X,Y)


#predict data for a single sample
y_pred_gauss=sc_y.inverse_transform(svr_obj.predict(sc_x.transform(np.array([[6.5]]))))
y_pred_poly=sc_y.inverse_transform(svr_poly.predict(sc_x.transform(np.array([[6.5]]))))
y_pred_sig=sc_y.inverse_transform(svr_sig.predict(sc_x.transform(np.array([[6.5]]))))


#plotting data 
plt.scatter(X,Y,color='red')
plt.plot(X,svr_obj.predict(X),color='blue')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.title('SVR--GAUSSIAN--MODEL')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,svr_poly.predict(X),color='blue')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.title('SVR--POLYNOMIAL--MODEL')
plt.show()


plt.scatter(X,Y,color='red')
plt.plot(X,svr_sig.predict(X),color='blue')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.title('SVR--SIGMOID--MODEL')
plt.show()

