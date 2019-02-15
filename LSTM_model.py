import numpy as np
import pandas as pd

trainData= pd.read_csv('Google_Stock_Price_Train.csv')
train=trainData.iloc[:,1:4].values

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
sc= MinMaxScaler(feature_range=(0,1))
train= sc.fit_transform(train)
#X1, X2, X3 are input, Y1, Y2, Y3 are corresponding outputs 
trainX1=[]
trainY1=[]
for i in range(60, 1258):
    trainX1.append(train[i-60:i, 0])
    trainY1.append(train[i, 0])
trainX1=np.array(trainX1)
trainY1=np.array(trainY1)
trainY1=np.reshape(trainY1,(1198,1))
trainX2=[]
trainY2=[]
for i in range(60, 1258):
    trainX2.append(train[i-60:i, 1])
    trainY2.append(train[i, 1])
trainX2=np.array(trainX2)
trainY2=np.array(trainY2)
trainY2=np.reshape(trainY2,(1198,1))

trainX3=[]
trainY3=[]
for i in range(60, 1258):
    trainX3.append(train[i-60:i, 2])
    trainY3.append(train[i, 2])
trainX3=np.array(trainX3)
trainY3=np.array(trainY3)
trainY3=np.reshape(trainY3,(1198,1))
totalY= np.stack((trainY1,trainY2,trainY3) , axis=0)
totalX= np.stack((trainX1,trainX2,trainX3) , axis=0)
totalX=totalX.reshape(1198,60,3)
totalY=np.transpose(totalY)
totalY=np.squeeze(totalY)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#window of 60 days used
rgs= Sequential()
rgs.add(LSTM(units=50, return_sequences=True, input_shape=(60, 3)))
rgs.add(Dropout(rate=0.2))

rgs.add(LSTM(units=50, return_sequences=True))
rgs.add(Dropout(rate=0.2))

rgs.add(LSTM(units=50, return_sequences=True))
rgs.add(Dropout(rate=0.2))

rgs.add(LSTM(units=50, return_sequences=True))
rgs.add(Dropout(rate=0.2))

rgs.add(LSTM(units=50))
rgs.add(Dropout(rate=0.2))

rgs.add(Dense(units=3))

rgs.compile(optimizer = 'adam', loss = 'mean_squared_error')

rgs.fit(totalX, totalY, epochs=100, batch_size=32)


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real = dataset_test.iloc[:, 1:4].values
real=np.array(real)
real=sc.transform(real)
finalTotal=np.concatenate([train, real])
inputs= finalTotal[len(finalTotal)-len(real)-60:]
## 1198,60,3

rgs.save('File.H5')
rgs.save_weights('file2.txt')



trainX1=[]
trainY1=[]
for i in range(60, 80): #from date 60 to 80, values set here just as example to demonstrate significance,
			#actual data size while training was much larger
    trainX1.append(inputs[i-60:i, 0])         
    trainY1.append(inputs[i, 0])
trainX1=np.array(trainX1)
trainY1=np.array(trainY1)

trainX2=[]
trainY2=[]
for i in range(60, 80):
    trainX2.append(inputs[i-60:i, 1])
    trainY2.append(inputs[i, 1])
trainX2=np.array(trainX2)
trainY2=np.array(trainY2)

trainX3=[]
trainY3=[]
for i in range(60, 80):
    trainX3.append(inputs[i-60:i, 2])
    trainY3.append(inputs[i, 2])
trainX3=np.array(trainX3)
trainY3=np.array(trainY3)

input= np.stack((trainX1,trainX2,trainX3) , axis=2)
print(np.shape(input))
input=np.reshape(input,(20,60,3))
predic=rgs.predict(input)
print (predic)
import matplotlib.pyplot as plt
predic= sc.inverse_transform(predic)
real= sc.inverse_transform(real)
plt.plot(real[:,0], color='red')

plt.plot(predic[:,0], color='green')
plt.show()