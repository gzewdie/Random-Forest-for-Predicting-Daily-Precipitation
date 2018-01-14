"""In this script we  apply the Random Forest machine learning   method with
  ECMWF data to predict Precipitation at a  location near Dallas, Texas (latitude:
  32.8 and Longitude : -97").

This is just a simple code to apply  the random Forest method. I will publish a Jupiter notebook
repository  showing further data analysis and implimenting other machine learning methods.

The Data is in mat format.
  """


# load the mat-file

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Read the mat file 
mat=h5py.File('ECMWFLSP4Pydata.mat','r')

#Extract  predictor and predictand variables 
X=mat['InRain']
Y=mat['OutRain']

# Names of Predictors  and the shape of the traing and output data
names=mat['usednames']
print(X.shape)
print(Y.shape)
#print(type(X))
#print(type(names))

# Transpose the matrix and shange the shape of Y [ravel]
X=np.transpose(X)
Y=np.transpose(Y)
Y=np.ravel(Y)

# The shape  of  the data the Random Forest  wants 
print(X.shape)
print(Y.shape)

# Plot and see the pollen Data
fig1=plt.figure()
plt.plot(Y,label="actual")
#plt.show()

"""Importing the mat file done!  Now let me apply  the Random Forest  regression"""

#Import the random Forest libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression 

#Apply the random Forest
regr = RandomForestRegressor(max_depth=100, random_state=0)
mdl=regr.fit(X,Y)

# Prediting  using the Same predictors data  we used in the traning 
ytest=mdl.predict(X)

"""Calculate the correlation coefficient between the  supervising
variable and the predicted variable"""

cc=np.corrcoef(Y,ytest)

print('The correlation coefficeint between the predicted  precipitation and the  supervising precipitations is: ' "%.2f" %  cc[0,1])
      
#print(ytest.shape)
#fig2=plt.figure()
plt.plot(ytest, label="predicted") 
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.title('Random Forest estimate, R=' "%.2f" %  cc[0,1])
plt.savefig('RFpredict.png')
plt.xlabel('Day of year')
plt.ylabel('Precipitation(m)')
plt.savefig('RFTrainng')
plt.show()


# Let us now apply apply Cross Validation 

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#predict and test the training 
mdl=regr.fit(X_train,y_train)
ypr=mdl.predict(X_train)

cctr=np.corrcoef(ypr,y_train)
#print(cctr)

#predict and test for the validation data
ytestR=mdl.predict(X_test)
ccvR=np.corrcoef(ytestR,y_test)
#print(ccvR)
#fig3=plt.figure() 
#scatter plot of the validation and traing
plt.plot([0, 0.006], [0,0.006]) # the  line passing through [0,0] and [1,1] 
plt.scatter(y_train, ypr, c="b", alpha=0.5,label='training')
plt.scatter(ytestR, y_test, c="r", alpha=0.5, label='validation')
plt.xlim( (0, 0.006) )
plt.ylim((0,0.006))
plt.title('Indipendent Validation, R=' "%.2f" %  ccvR[0,1])
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel('actual precipitation(m)')
plt.ylabel('predicted precipitation (m)')
plt.savefig('RFscatterplot.png')
plt.show()

#line plot for cross-validation
plt.figure
plt.plot(y_test,label='actual precipitation(m)')
plt.plot(ytestR, label="predicted precipitation(m)")
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.title('Random Forest Validation , R=' "%.2f" %  ccvR[0,1])
plt.xlabel('actual precipitation(m)')
plt.ylabel('predicted precipitation (m)')
plt.savefig('RFValidation.png')
plt.show()


