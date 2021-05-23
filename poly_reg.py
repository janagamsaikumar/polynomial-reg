import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv(r'D:\data science\AMXWAM data science\class 18_oct 18,2020\6.POLYNOMIAL REGRESSION\Position_Salaries.csv')
# dataset contains employee experience and salary 
# we have 10 different positions and positions salaries
# company hiring for new employee anad the person who is hired he has 20+ year exp hr team has to verify his details with the previous company
# the hired expployee  previous salary was 161k and now expecting more than 161k
#hr team got the details of the employee and he has 2 years of experience as a regional manager
# it takes 4 year to jump regional to partner and he is halfway to get into partner.
# using polynomial regression model we are going to find the salalry of this employee and  predict his future salary going to be 
X=dataset.iloc[:,1:-1]
y=dataset.iloc[:,-1]
  # check there any null values
X.notnull().any()
y.notnull().any()
 # it is a small dataset  with only 10 observations we dont need to train the data set just have to predict manually my developing poly algo.

 # X=X.drop(columns='Position',axis=1)  we have dropped the positions column because we have levels
# which is highly  corelated with the dependent varibale
# always check attribute which is highly corelated with the dependent variable

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,y)
#
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lr2=LinearRegression()
lr2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,y,color='red')
plt.plot(X,lr2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
 # till here we have created a polynomial regression model 
 # where for degree 2 to 3 my acatual points are not touching the predicted line which result incorrect model
 # here degree 4 touches my all the data points i.e actual data and predicted data are connected
 # polynomial is the combination of slr and degrees
 
 # now have to predict the 6.5 years experience persons salary
reg.predict([[6.5]]) 
# if u check this my linear regression model predicted that 6.5 exp employee will get 330k per annum
 
# predicting with polynimial regression
Y=lr2.predict(poly_reg.fit_transform([[6.5]])) # my model has predicted that the employee salary is 158k so we can the perosn is a genuine.
