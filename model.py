
import sys
import os
import argparse
import pandas
import cv2
import numpy as np

import sklearn.utils

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Cropping2D, MaxPooling2D, Dropout, Dense, Flatten

class training_data:
    def __init__(self, data_path, valid_percent):
        self.full_images = []
        self.full_steering = []
        self.valid_percent =  valid_percent
    
        self.add_paths(data_path)
        
    def add(self, path):
        data_path = os.path.abspath(path)
        print ("Reading path:", data_path)
        self.csv_data = pandas.read_csv(os.path.join(data_path, 'driving_log.csv'), header=None)
        
        image_sets = self.csv_data[[0, 1, 2]].values
        steering_angles  = self.csv_data[3].values   
        
        
        for image_set, steering_angle in zip(image_sets, steering_angles):
            self.full_images += [self.get_real_path(data_path, image_set[0]), self.get_real_path(data_path, image_set[1]), self.get_real_path(data_path, image_set[2])]
            self.full_steering += [steering_angle, steering_angle + 0.2, steering_angle - 0.2]
            
    def get_real_path(self,directory, path):
        path_parts = path.split("\\")        
        real_path = os.path.join(directory, path_parts[-2], path_parts[-1])
        return real_path
                                 
    def add_paths(self, paths):
        paths = paths.split(";")
        
        for path in paths:
            self.add(path)
        
        self.done_adding()
        
    def done_adding(self):
        self.train, self.valid, self.train_steering, self.valid_steering = train_test_split(self.full_images, self.full_steering, test_size=self.valid_percent/100, random_state=2017)
        
        print("Training set size is:", len(self.train))
        print("Validation set size is:", len(self.valid))

    
    def image_read(self,path):
        image =  cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def random_brightness(self,image):
        new_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = 0.5 + 0.5*(2*np.random.rand()-1.0)    
        new_image[:,:,2] = new_image[:,:,2]*random_bright
        image1 = cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)
        return image1
    
    def skew_position(self, img,steering, range_x = 100, range_y = 20):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        new_steering = steering + trans_x * 0.004
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = img.shape[:2]
        new_image = cv2.warpAffine(img, trans_m, (width, height))
        return new_image, new_steering
    
    def generate_helper(self, image_set, steering_set, skew, batch_size):
        num_samples = len(image_set)
        while 1:
            shuffle_images, shuffle_steering = sklearn.utils.shuffle(image_set, steering_set)
            for offset in range(0, num_samples, batch_size):
                image_batch = shuffle_images[offset:offset+batch_size]
                steering_batch = shuffle_steering[offset:offset+batch_size]
    
                images = []
                steering_angles = []
                for path, steering_angle in zip(image_batch, steering_batch):
                    image = self.image_read(path)                    
                    
                    if np.random.randint(0,2)==0:
                        image,steering_angle = cv2.flip(image,1),-steering_angle                    
                    
                    new_image = self.random_brightness(image)
                    
                    new_steering = steering_angle
                    if (skew == True):
                        new_image, new_steering = self.skew_position(new_image, steering_angle) 
                    
                    images += [new_image]
                    steering_angles += [new_steering]
    
                yield sklearn.utils.shuffle(np.array(images), np.array(steering_angles))
    
    def generate_training_batch(self, batch_size):
        return self.generate_helper(image_set=self.train, steering_set=self.train_steering, skew = True, batch_size=batch_size)
                
    def generate_validation_batch(self, batch_size):
        return self.generate_helper(image_set=self.valid, steering_set=self.valid_steering, skew = False, batch_size=batch_size)


def get_keras_model():
    
    model = Sequential()
    model.add(Lambda(lambda x: x/255-1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))    
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,activation='tanh'))
    model.summary()      

    return model

def train_model(model, data):
    
    checkpoint = ModelCheckpoint('out/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
    
    model.fit_generator(data.generate_training_batch(200), samples_per_epoch = 9000,
                        nb_epoch=20, verbose=1, callbacks=[checkpoint],
                        validation_data=data.generate_validation_batch(500),nb_val_samples = 500)

def main():

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Training the model to drive.')
    parser.add_argument("-d", "--data-dir", dest="data_dir", help="Training data Path")
    parser.add_argument("-v", "--validation-percent", dest="valid_percent", help="Percent of data to be used for the validation")

    # Process arguments
    args = parser.parse_args()
    
    training_data_path = "data"
    if args.data_dir:
        training_data_path = args.data_dir
        
    
    validation_percent = 20
    if args.valid_percent:
        validation_percent = args.valid_percent
    
    data = training_data(training_data_path, validation_percent)
    
    batch, values = next(data.generate_training_batch(10))
     
    print(batch.shape, values.shape)
    
    model = get_keras_model()
     
    train_model(model, data) 
    


if __name__ == "__main__":
    sys.exit(main())