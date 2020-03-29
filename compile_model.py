from modelcofig import modelconfig
import tensorflow.keras as k

def compile_model_adam(model,lr):
	opt=k.optimizers.Adam(lr)
	model.compile(optimizer=opt,loss=k.losses.categorical_crossentropy,metrics=['accuracy'])
	return model
