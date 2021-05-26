from tensorflow import keras

def get_model(num_detectors=1, author='gabbard', freq_data=False):
    author_dict = {'gabbard': get_model_gabbard,
                   'george': get_model_george}
    return author_dict[author.lower()](num_detectors=num_detectors,
                                       freq_data=freq_data)

def get_model_george(num_detectors=1, freq_data=False):
    if freq_data:
        inp = keras.layers.Input(shape=(1025, 2*num_detectors), name='Input')
    else:
        inp = keras.layers.Input(shape=(2048, num_detectors), name='Input')

    bn = keras.layers.BatchNormalization()(inp)
    conv1 = keras.layers.Conv1D(16, 16)(bn)
    pool1 = keras.layers.MaxPooling1D(4, strides=4)(conv1)
    act1 = keras.layers.Activation('relu')(pool1)

    conv2 = keras.layers.Conv1D(32, 8, dilation_rate=4)(act1)
    pool2 = keras.layers.MaxPooling1D(4, strides=4)(conv2)
    act2 = keras.layers.Activation('relu')(pool2)

    conv3 = keras.layers.Conv1D(64, 8, dilation_rate=4)(act2)
    pool3 = keras.layers.MaxPooling1D(4, strides=4)(conv3)
    act3 = keras.layers.Activation('relu')(pool3)

    flat = keras.layers.Flatten()(act3)
    dense1 = keras.layers.Dense(64, activation='relu')(flat)
    dense2 = keras.layers.Dense(2, activation='softmax')(dense1)

    model = keras.models.Model(inputs=[inp], outputs=[dense2])
    return model

def get_model_gabbard(num_detectors=1, freq_data=False):
    #Network Hunter (Adjusted Pool-sizes due to other sample-rate)
    if freq_data:
        inp = keras.layers.Input(shape=(1025, 2*num_detectors))
    else:
        inp = keras.layers.Input(shape=(2048, num_detectors))
    b1 = keras.layers.BatchNormalization()(inp)
    c1 = keras.layers.Conv1D(8, 64)(b1)
    a1 = keras.layers.Activation('elu')(c1)
    c2 = keras.layers.Conv1D(8, 32)(a1)
    p1 = keras.layers.MaxPooling1D(4)(c2)
    a2 = keras.layers.Activation('elu')(p1)
    c3 = keras.layers.Conv1D(16, 32)(a2)
    a3 = keras.layers.Activation('elu')(c3)
    c4 = keras.layers.Conv1D(16, 16)(a3)
    p2 = keras.layers.MaxPooling1D(3)(c4)
    a4 = keras.layers.Activation('elu')(p2)
    c5 = keras.layers.Conv1D(32, 16)(a4)
    a5 = keras.layers.Activation('elu')(c5)
    c6 = keras.layers.Conv1D(32, 16)(a5)
    p3 = keras.layers.MaxPooling1D(2)(c6)
    a6 = keras.layers.Activation('elu')(p3)
    f1 = keras.layers.Flatten()(a6)
    d1 = keras.layers.Dense(64)(f1)
    dr1 = keras.layers.Dropout(0.5)(d1)
    a7 = keras.layers.Activation('elu')(dr1)
    d2 = keras.layers.Dense(64)(a7)
    dr2 = keras.layers.Dropout(0.5)(d2)
    a8 = keras.layers.Activation('elu')(dr2)
    d3 = keras.layers.Dense(2, activation='softmax')(a8)
    
    model = keras.models.Model(inputs=[inp], outputs=[d3])
    return model
