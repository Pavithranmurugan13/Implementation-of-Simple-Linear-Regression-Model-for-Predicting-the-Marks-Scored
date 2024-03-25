# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Predict the regression for marks by using the representation of the graph
4. Compare the graphs and hence we obtained the linear regression for the given datas 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: pavithran m j
RegisterNumber:  212223240112
*/
```
```import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)```
## Output:
![simple linear regression model for predicting the marks scored](sam.png)
##Dataset:
![Screenshot 2024-03-25 082534](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/dc161538-02f1-4b79-a632-9783951800ad)
##Head values:
![Screenshot 2024-03-25 082552](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/f069b9b3-e072-426f-921b-9cb99ba7adac)
##Tail values:
![Screenshot 2024-03-25 082600](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/d853d680-b2e5-4669-ab73-2cce8b85e349)
##X and Y values:
![Screenshot 2024-03-25 082616](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/72c7c1c5-2bed-405b-b8b4-9840db5e9d1f)
##predication value of x and y:
![Screenshot 2024-03-25 082637](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/6fea6303-9914-45e0-af1e-6e947172110c)
##MSE,MAE and RMSE:
![Screenshot 2024-03-25 082646](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/3b804304-ec97-46db-a50b-b68b4137e1d1)
##Training Set:
![Screenshot 2024-03-25 082715](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/7e620b97-83dc-466d-bd50-bd245b02909c)
##Testing Set:
![Screenshot 2024-03-25 082729](https://github.com/Pavithranmurugan13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/163802201/8e5d49ee-081c-4a2a-88fe-b6ac285ec540)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
