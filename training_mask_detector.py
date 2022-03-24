from keras.applications.vgg19 import VGG19
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2



directory = 'C:/Users/33616/Desktop/Face-Mask-Detection/dataset/'
EPOCHS = 6
BATCHSIZE = 32


datagen = ImageDataGenerator(rescale=1.0 / 255,
                             shear_range=0.25,
                             horizontal_flip=True,
                             zoom_range=0.3,
                             vertical_flip=True,
                             validation_split=0.2
                             )


train_generator = datagen.flow_from_directory(directory=directory,
                                              target_size=(128,128),
                                              subset='training',
                                              class_mode='categorical',
                                              batch_size=32)


model1 = Sequential([
    MobileNetV2(input_shape=(128,128,3),weights="imagenet",include_top=False),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(units=3, activation='softmax')
])

model1.summary()

optimizer = keras.optimizers.Adam()

model1.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics =['accuracy'])


r1 = model1.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCHSIZE,
    epochs = EPOCHS)

model1.save("MaskModel.h5")
