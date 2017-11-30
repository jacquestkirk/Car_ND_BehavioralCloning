import cv2
import sklearn
import numpy as np

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = "TrainingData\\IMG\\" + batch_sample.imageLink.split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = batch_sample.steering

                if(batch_sample.flip):
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                else:
                    images.append(center_image)
                    angles.append(center_angle)



            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)