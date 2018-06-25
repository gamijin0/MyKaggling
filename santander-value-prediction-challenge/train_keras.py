from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD,RMSprop


model = Sequential()
model.add(Dense(units=512,input_dim=4991,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.1)
model.compile(optimizer='adam',loss='mse',metrics=['mae'])





import pandas as pd

train = pd.read_csv("data/train.csv",header=0)
# test = pd.read_csv("data/test.csv",header=0)

y_train = train['target']
x_train = train.iloc[:,2:]

model.fit(x=x_train,y=y_train,verbose=1,epochs=1000)