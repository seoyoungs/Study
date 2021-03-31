# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    # urllib.request.urlretrieve(url, 'rps.zip')
    # local_zip = 'rps.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('C:/data/image/')
    # zip_ref.close()

    TRAINING_DIR = "C:/data/image/rps/"
    train_datagen = ImageDataGenerator(
        rescale = 1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size = 32,
        class_mode='categorical',
        target_size = (150, 150),
        subset = 'training'
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )

    validation_generator = validation_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size = 32,
        class_mode='categorical',
        target_size = (150, 150),
        subset = 'validation'
    )


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (2,2), padding='same', 
        activation = 'relu', input_shape=(150, 150, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(3,3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') # 다중분류
    ])

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['acc'])

    # history= model.fit_generator(
    # train_generator, steps_per_epoch=32, epochs=20,
    # validation_data=validation_generator
    # )
    history = model.fit(train_generator, steps_per_epoch=8, epochs=40, 
            verbose=1, validation_data=validation_generator, validation_steps=8
    )

    loss, acc = model.evaluate(validation_generator)
    print("loss : ", loss)
    print("acc : ", acc)

    return model

'''
loss :  0.7509967684745789
acc :  0.7579365372657776
'''

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category3_1/mymodel.h5")
