import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout
from keras.models import Sequential

import imageGenerator



speedThreshold = 1 #ignore lines if the speed is <10. Steering angle has little effect when moving very slowly.
batchSize = 32
testSize_ratio = 0.2
inputImageSize = (160,320,3)
cropAmount_px_t_b_l_r = [50,20,0,0]
num_epochs = 10
steeringCorrection = 0.2
flipMinAngle = 0.75



def ModelBuilder():
    model = Sequential()

    #cropping
    model.add(Cropping2D(cropping=((cropAmount_px_t_b_l_r[0],cropAmount_px_t_b_l_r[1]),
                                   (cropAmount_px_t_b_l_r[2],cropAmount_px_t_b_l_r[3]))
                         ,input_shape=inputImageSize))

    #calculate new image size
    croppedImageSize = (inputImageSize[0] - cropAmount_px_t_b_l_r[0] - cropAmount_px_t_b_l_r[1],
                       inputImageSize[1] -cropAmount_px_t_b_l_r[2] -cropAmount_px_t_b_l_r[3],
                       inputImageSize[2])

    #pre-processing
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=croppedImageSize,
                     output_shape=croppedImageSize))

    #Layers
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(Flatten(input_shape=croppedImageSize))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(84, activation='relu'))


    #output layer
    model.add(Dense(1))

    #optimizer
    model.compile(loss='mse', optimizer='adam')

    return(model)


class SteeringData:
    def __init__(self, imageLink, steering, flip):
        self.imageLink = imageLink
        self.steering = steering
        self.flip = flip





dataList = []

print('loading file')
with open('TrainingData//driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        if float(line[6]) > float(speedThreshold):    #only append high velocity lies

            #pull out steering angles
            steering = float(line[3])
            steeringLeft = float(steering) + steeringCorrection
            steeringRight = float(steering) - steeringCorrection

            #pull out steering
            imageCenter = line[0]
            imageLeft = line[1]
            imageRight = line[2]

            #add new data to the list
            dataList.append(SteeringData(imageCenter, steering, False))
            dataList.append(SteeringData(imageLeft, steeringLeft, False))
            dataList.append(SteeringData(imageRight, steeringRight, False))

            #add flipped versions
            if(abs(steering) > flipMinAngle):
                dataList.append(SteeringData(imageCenter, steering, True))
                dataList.append(SteeringData(imageLeft, steeringLeft, True))
                dataList.append(SteeringData(imageRight, steeringRight, True))



train_samples, validation_samples = train_test_split(dataList, test_size=testSize_ratio)

train_generator = imageGenerator.generator(train_samples, batch_size=batchSize)
validation_generator = imageGenerator.generator(validation_samples, batch_size=batchSize)

print('building model')
model = ModelBuilder()

print('training')
history_object = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=num_epochs)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print('saving model')
model.save('model.h5')