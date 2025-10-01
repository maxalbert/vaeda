#- classifier
import tensorflow as tf
import tf_keras as tfk
import numpy as np
    

    
    
def define_classifier(ngens, seed=1, num_layers=1): 
    
    tfkl = tfk.layers
    
    if(num_layers==1):
        classifier = tfk.Sequential([
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.BatchNormalization(),
            tfkl.Dense(1, activation='sigmoid')
        ])
    if(num_layers==2):
        print('using 2 layers in classifier')
        classifier = tfk.Sequential([
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.BatchNormalization(),
            tfkl.Dense(3, activation='relu'),
            tfkl.Dense(1, activation='sigmoid')
        ])

    model = tfk.Model(inputs=classifier.inputs,
                      outputs=classifier.outputs[0])

    return model
    
    
