import tensorflow.keras as k
from preprocess import preprocess

def load_and_augment_data(path,tpath):
    images=k.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=None,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=preprocess,
    data_format=None,
    validation_split=0.05,
    dtype=None)

    training_gen=images.flow_from_directory(path,
    target_size=(75,75),
    color_mode='grayscale',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='ppm',
    follow_links=False,
    subset='training',
    interpolation='nearest')


    testing_gen=images.flow_from_directory(tpath,
    target_size=(75,75),
    color_mode='grayscale',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='ppm',
    follow_links=False,
    subset='validation',
    interpolation='nearest')
    return training_gen,testing_gen
  	