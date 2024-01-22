
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet import MobileNet
import PIL
import os
import shutil

IMAGE_DIR = 'images'

image_generator = ImageDataGenerator(validation_split=0.2)  

train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='./images',
                                                 target_size=(160, 160), 
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='./images',
                                                 target_size=(160, 160), 
                                                 subset="validation",
                                                 class_mode='categorical')

learning_rate = 0.01
layer_size = 256
drop_rate = 0.2
scores = {}

filepath = './mobilent_v2_{epoch:02d}_{val_accuracy:.3f}.h5'
    


def get_data():
    

    os.system("rm -rf ./images")
    os.mkdir(IMAGE_DIR)
    os.system("wget -q -O ./images/daisy.zip https://github.com/PriyaVellanki/flower_classification/raw/main/data/daisy.zip")
    os.system("wget -q -O ./images/dandelion.zip https://github.com/PriyaVellanki/flower_classification/raw/main/data/dandelion.zip")
    os.system("unzip ./images/daisy.zip -d ./images/")
    os.system("unzip ./images/dandelion.zip -d ./images/")
    shutil.rmtree("./images/__MACOSX",ignore_errors=True)

def get_model():
    base_model=MobileNet(input_shape=(160, 160, 3),
                     weights='imagenet',
                     include_top=False
                    )
    base_model.trainable = False
    inputs = keras.Input(shape=(160, 160, 3))

    base = base_model(inputs, training=False)

    vectors = keras.layers.GlobalAveragePooling2D()(base)

    outputs = keras.layers.Dense(2)(vectors)

    model = keras.Model(inputs, outputs)

    return model


def mobilenet_model(learning_rate=0.01,size=1024,drop_rate=0.2):

    base_model=MobileNet(input_shape=(160, 160, 3),
                     weights='imagenet',
                     include_top=False
                    )
    base_model.trainable = False


    inputs = keras.Input(shape=(160, 160, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size,activation='relu')(vectors)
    dropout = keras.layers.Dropout(drop_rate)(inner)
    outputs = keras.layers.Dense(2)(dropout)
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model



if __name__ == '__main__':
    get_data() 
    mobilenet_model(learning_rate=0.01,size=1024,drop_rate=0.2)   

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                             save_best_only=True,
                                             monitor='val_accuracy',
                                             mode='max')

    model = mobilenet_model(learning_rate=learning_rate, size=layer_size,drop_rate=drop_rate)
    history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset,callbacks=[checkpoint])

