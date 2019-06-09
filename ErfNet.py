import keras

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import ReLU
from keras.layers import MaxPool2D
from keras.layers import Conv2DTranspose
from keras.layers import add
from keras.layers import concatenate
from keras.layers import Input

from keras.models import Model
from keras.optimizers import Nadam, Adam

import numpy as np

keras.backend.image_data_format('channels_last')

class ERFNet:
    #================================ INIT ===============================================
    def __init__(self, shape):
        self.input_shape  =  shape
        self.trainMode    =  True
        self.model        =  self._get_model()

    #================================ GET MODEL ==========================================
    def _get_model(self, verbose = True):
        model = {}

        model['input'] = Input(shape=self.input_shape)
        
        # encoder
        model['down_1_main'] = self._downsampler_block(model, 'input', 16, 1)
        model['down_2_main'] = self._downsampler_block(model, 'down_1_main', 64, 2)

        model['btnk_3_main'] = self._non_bottleneck_1d(model, 'down_2_main', 64, 0.03, 1, 3 )
        model['btnk_4_main'] = self._non_bottleneck_1d(model, 'btnk_3_main', 64, 0.03, 1, 4 )
        model['btnk_5_main'] = self._non_bottleneck_1d(model, 'btnk_4_main', 64, 0.03, 1, 5 )
        model['btnk_6_main'] = self._non_bottleneck_1d(model, 'btnk_5_main', 64, 0.03, 1, 6 )

        model['down_7_main'] = self._downsampler_block(model, 'btnk_6_main', 128, 7)

        model['btnk_8_main'] = self._non_bottleneck_1d(model,  'down_7_main',  128, 0.03, 2, 8 )
        model['btnk_9_main'] = self._non_bottleneck_1d(model,  'btnk_8_main',  128, 0.03, 4, 9 )
        model['btnk_10_main'] = self._non_bottleneck_1d(model, 'btnk_9_main',  128, 0.03, 8, 10 )
        model['btnk_11_main'] = self._non_bottleneck_1d(model, 'btnk_10_main', 128, 0.03, 16, 11 )
        model['btnk_12_main'] = self._non_bottleneck_1d(model, 'btnk_11_main', 128, 0.03, 2, 12 )
        model['btnk_13_main'] = self._non_bottleneck_1d(model, 'btnk_12_main', 128, 0.03, 4, 13 )
        model['btnk_14_main'] = self._non_bottleneck_1d(model, 'btnk_13_main', 128, 0.03, 8, 14 )
        model['btnk_15_main'] = self._non_bottleneck_1d(model, 'btnk_14_main', 128, 0.03, 16, 15 )

        # decoder
        model['up_16_main']   = self._upsampler_block(model,   'btnk_15_main', 64, 2, 16)
        model['btnk_17_main'] = self._non_bottleneck_1d(model, 'up_16_main',   64, 0, 1, 17 )
        model['btnk_18_main'] = self._non_bottleneck_1d(model, 'btnk_17_main', 64, 0, 1, 18 )
       
        model['up_19_main']   = self._upsampler_block(model,   'btnk_18_main', 16, 2, 19)
        model['btnk_20_main'] = self._non_bottleneck_1d(model, 'up_19_main',   16, 0, 1, 20 )
        model['btnk_21_main'] = self._non_bottleneck_1d(model, 'btnk_20_main', 16, 0, 1, 21 )

        model['output'] = Conv2DTranspose(1 ,(2,2), strides=2, padding = 'valid',activation='relu' )( model['btnk_21_main'])

        # define model
        net = Model(inputs = model['input'], outputs =  model['output'])

        # choose optimizer
        # optim  = Adam(lr = 0.01, beta_1 =0.5, beta_2=0.99, decay=0.001)
        optim = Nadam(lr = 0.01, beta_1 =0.9, beta_2=0.99, schedule_decay = 0.001)
        
        # compile model
        net.compile(optimizer = optim, loss = 'mse', metrics = ['accuracy'])

        if verbose:
            net.summary()

        return net


    #================================ BOTTLENECK =========================================
    def _non_bottleneck_1d(self, model, lastLayer, noFilters, dropRrate, dilRate, moduleId ):
        modId = str(moduleId) + '_btnk'

        model['conv3x1_1_'  + modId]  = Conv2D(noFilters, (3, 1), 
                                                strides = 1, 
                                                padding = 'same'
                                            )(model[lastLayer])
        model['relu_1_1_'   + modId]  = ReLU()(model['conv3x1_1_' + modId] )

        model['conv1x3_1_'  + modId]  = Conv2D(noFilters, (3, 1), 
                                                strides = 1, 
                                                padding = 'same'
                                            )(model['relu_1_1_' + modId])

        model['batch_norm_1_' + modId]= BatchNormalization(
                                                epsilon = 1e-03
                                            )(model['conv1x3_1_' + modId])
        
        model['relu_1_2_'   + modId]  = ReLU()(model['batch_norm_1_' + modId] )
        
        model['conv3x1_2_'  + modId]  = Conv2D(noFilters, (3, 1), 
                                                strides = 1, 
                                                padding = 'same', 
                                                dilation_rate = (dilRate, 1)
                                            )(model['relu_1_2_' + modId])
        
        model['relu_2_1_'   + modId]  = ReLU()(model['conv3x1_2_' + modId] )
        
        model['conv1x3_2_'  + modId]  = Conv2D(noFilters, (3, 1), 
                                                strides = 1, 
                                                padding = 'same', 
                                                dilation_rate = (1, dilRate)
                                            )(model['relu_2_1_' + modId])
        
        model['batch_norm_2_'  + modId]  = BatchNormalization(
                                                epsilon = 1e-03
                                                    )(model['conv1x3_2_' + modId])
        

        #if the model is in train mode, a Dropout layer is added
        if self.trainMode :
            model['drop_1_' + modId] = Dropout(
                                            rate = dropRrate
                                        )(model['batch_norm_2_'  + modId])
            model['add_'    + modId]    = add([model['drop_1_' + modId], model[lastLayer]])
        else:
            model['add_' + modId]    = add([model['batch_norm_2_'  + modId], model[lastLayer]])

        return ReLU()(model['add_' + modId] ) 

    #================================ DOWNSAMPLER ========================================
    def _downsampler_block(self, model, lastLayer, noFilters, moduleId):
        modId = str(moduleId) + '_down'

        # compute the output no
        shape = model[lastLayer].get_shape()
        outNo = int(shape[-1])

        model['conv_1_' + modId] = Conv2D( filters = noFilters - outNo , kernel_size = (2,2), strides=2, padding='valid')(model[lastLayer])
        model['pool_1_' + modId] = MaxPool2D(2, 2)(model[lastLayer])

        model['concat_1_' + modId] = concatenate([model['pool_1_' + modId], model['conv_1_' + modId]])
        model['batch_norm_1' + modId] = BatchNormalization(epsilon=1e-03)(model['concat_1_' + modId])
        
        return ReLU()( model['batch_norm_1' + modId])

    #================================ UPSAMPLER ==========================================
    def _upsampler_block(self, model, lastLayer, noFilters,strides ,moduleId, ):
        modId = str(moduleId) + '_up'

        model['deconv_1_'+ modId] = Conv2DTranspose(noFilters, (3,3), strides = strides, padding = 'same')(model[lastLayer])
        model['batch_norm_1_' + modId] = BatchNormalization(epsilon=1e-03)(model['deconv_1_'+ modId])
       
        return ReLU()(model['batch_norm_1_' + modId])
