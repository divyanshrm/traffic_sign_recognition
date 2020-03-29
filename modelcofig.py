from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Input,Convolution2D,MaxPool2D,Flatten


def modelconfig(dropout_rate)
	model=Sequential()
	model.add(Input(shape=(75,75,1)))
	model.add(Convolution2D(64,(3,3),activation='relu'))
	model.add(MaxPool2D((2,2)))
	model.add(Convolution2D(64,(3,3),activation='relu'))
	model.add(MaxPool2D((2,2)))
	model.add(Convolution2D(128,(3,3),activation='relu'))
	model.add(MaxPool2D((2,2)))
	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(43,activation=k.activations.softmax))
	return model