# This script is to apply machine learnin  to the ECMWF data

#To load mat-file

import h5py
import numpy as np
import matplotlib.pyplot as plt

mat=h5py.File('ECMWFLSP4Pydata.mat','r')


X=mat['InRain']
Y=mat['OutRain']
names=mat['usednames']
print(X.shape)
print(Y.shape)
print(type(X))
print(type(names))

# Transpose the matrix and shange the shape of Y [ravel]
X=np.transpose(X)
Y=np.transpose(Y)
Y=np.ravel(Y)
print(X.shape)
print(Y.shape)

# Plot and see the pollen Data
plt.plot(Y,label="actual")
#plt.show()

# Importing the mat file works!

# Now let me apply  The Random Forest  regression

from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression 

#Apply the random Forest
regr = RandomForestRegressor(max_depth=100, random_state=0)
mdl=regr.fit(X,Y)

ytest=mdl.predict(X)

cc=np.corrcoef(Y,ytest)
print(cc)
print(ytest.shape)
plt.figure 
plt.plot(ytest, label="predicted")
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.title('Random Forest estimate, R=' "%.2f" %  cc[0,1])
plt.savefig('RFpredict.png')
plt.show()


# Apply Cross Validation 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# predict and test the training 
mdl=regr.fit(X_train,y_train)
ypr=mdl.predict(X_train)

cctr=np.corrcoef(ypr,y_train)
print(cctr)

#predict and test for the validation data
ytestR=mdl.predict(X_test)
ccvR=np.corrcoef(ytestR,y_test)
print(ccvR)
plt.figure 
#scatter plot of the validation and traing
plt.plot([0, 0.006], [0,0.006])
plt.scatter(y_train, ypr, c="b", alpha=0.5,label='training')
plt.scatter(ytestR, y_test, c="r", alpha=0.5, label='validation')
plt.xlim( (0, 0.006) )
plt.ylim((0,0.006))
plt.title('Indipendent Validation, R=' "%.2f" %  ccvR[0,1])
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.savefig('RFscatterplot.png')
plt.show()


plt.figure
plt.plot(y_test,label='actual')
plt.plot(ytestR, label="predicted")
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.title('Random Forest Validation , R=' "%.2f" %  ccvR[0,1])
plt.savefig('RFValidation.png')
plt.show()


print(Y.shape)
