import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from load_and_augment import load_and_augment_data
from modelconfig import modelconfig
from compile_model import compile_model_adam

path=r'D:\data_traffic\Training'
testing_path=r'D:\data_traffic\Testing'
training_gen,testing_gen=load_and_augment_data(path,testing_path)
model=modelconfig(0.33)
model=compile_model_adam(model,0.0001)
cb=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
model.fit_generator(generator=training_gen,steps_per_epoch=50,epochs=50,validation_data=testing_gen, validation_steps=10,callbacks=[cb])
model.save('model_traffic.h5')