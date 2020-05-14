from skimage import exposure
import numpy as np
def preprocess(data):
    data=np.reshape(data,(75,75))
    data=exposure.equalize_hist(data)
    data=data/255
    data=np.reshape(data,(75,75,1))
    return data