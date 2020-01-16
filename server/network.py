import os
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose


def default_n_linear(num_outputs, input_shape=(120, 160, 3), roi_crop=(0, 0)):

    drop = 0.1
    
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Cropping2D(cropping=(roi_crop, (0,0)))(x) #trim pixels off top and bottom
    #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = BatchNormalization()(x)
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = []
    
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs)
    
    return model


class KerasLinear():
    def __init__(self, num_outputs=2, input_shape=(120, 160, 6), roi_crop=(0, 0), *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        self.optimizer = "adam"
        self.model = default_n_linear(num_outputs, input_shape, roi_crop)
        self.compile()
    
    def set_optimizer(self, rate, decay):
        self.model.optimizer = keras.optimizers.Adam(lr=rate, decay=decay)


    def train(self, train_gen, val_gen, 
              saved_model_path, epochs=100, train_steps=1260,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        
        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=verbose, 
                                                    save_best_only=True, 
                                                    save_weights_only=True,
                                                    mode='min')
        
        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=min_delta, 
                                                   patience=patience, 
                                                   verbose=verbose, 
                                                   mode='auto')
        
        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)
        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=train_steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list, 
                        validation_steps=50)
        return hist


    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def default_n_softmax(num_outputs, input_shape=(120, 160, 3), roi_crop=(0, 0)):

    drop = 0.1
    
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Cropping2D(cropping=(roi_crop, (0,0)))(x) #trim pixels off top and bottom
    #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = BatchNormalization()(x)
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = []
    
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='softmax', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs)
    
    return model

class KerasPolicy():
    def __init__(self, num_outputs=11, input_shape=(120, 160, 6), roi_crop=(0, 0), *args, **kwargs):
        super(KerasPolicy, self).__init__(*args, **kwargs)
        self.optimizer = "adam"
        self.num_outputs = num_outputs
        self.model = default_n_linear(num_outputs, input_shape, roi_crop)
        self.compile()

        self.states = []
        self.rewards = []
        self.actions = []
        self.probabilites = []
    
    def set_optimizer(self, rate=.001):
        self.model.optimizer = keras.optimizers.Adam(lr=rate)


    def train(self, saved_model_path, discount_rate=.99):
        
        # training
        running_reward = 0
        for reward, action, probabilites in zip(self.rewards, self.actions, self.probabilites)[::-1]:
            running_reward = running_reward * .99 + reward
            action_prob = np.zeros((self.num_outputs))
            action_prob[action] = np.log(probabilites[action])
            loss = -action_prob * running_reward
            # This updates per frame instead of per episode, might wanna do a lot more losses at once
            # https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
            self.model.optimizer.minimize(loss)

        # saving weights idk I don't wanna set up real checkpointing
        self.model.save_weights('saved_model_path')

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                loss='categorical_crossentropy')

    def run(self, img_arr, mode="training"):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        if(mode == "training"):
            action = np.random.choice(self.num_outputs, 1, p=outputs)[0]
        else:
            action = np.argmax(outputs)

        angle = (action - self.num_outputs // 2 - 1) / self.num_outputs * 25 + 90
        
        if(mode == "training"):
            self.probabilites.append(outputs)
            self.actions.append(action)
            self.rewards.append(1)

        return 80, angle
    
    def clear_buffers(self):
        self.rewards = []
        self.actions = []
        self.probabilites = []

class KerasLinearPolicy():
    def __init__(self, num_outputs=11, input_shape=(120, 160, 6), roi_crop=(0, 0), *args, **kwargs):
        super(KerasPolicy, self).__init__(*args, **kwargs)
        self.set_optimizer()
        self.model = default_n_softmax(num_outputs, input_shape, roi_crop)
        self.compile()
