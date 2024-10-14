import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import backend as K
from utils import loss_mae, loss_function


def residual_block(x, units):
    shortcut = x  # 保存输入
    x = Dense(units, activation='relu')(x)
    x = Dense(units, activation=None)(x)
    x = Add()([x, shortcut])  # 将输入与输出相加
    x = Activation('relu')(x)
    return x

# def forward_network(units_layer1=256, units_layer2=1600, units_layer3=1600, units_layer4=512, optimizer='rmsprop'):
#     inp = Input(shape=(9,), name='forward_input')
#     x = inp
#     x = Dense(units_layer1)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer2)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer3)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer4)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     out = Dense(20, activation=None)(x)
#     model = Model(inputs=inp, outputs=out)
#
#     if optimizer == 'sgd':
#         optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
#     elif optimizer == 'adam':
#         optimizer = Adam(learning_rate=0.001)
#     elif optimizer == 'rmsprop':
#         optimizer = RMSprop(learning_rate=0.001)
#
#     model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
#     return model

def forward_network(units_layer1=256, units_layer2=1600, units_layer3=1600, units_layer4=512, optimizer='rmsprop'):
    inp = Input(shape=(4,), name='forward_input')
    x = inp
    x = Dense(units_layer1, activation='relu')(x)
    x = residual_block(x, units_layer1)

    x = Dense(units_layer2, activation='relu')(x)
    x = residual_block(x, units_layer2)

    x = Dense(units_layer3, activation='relu')(x)
    x = residual_block(x, units_layer3)

    x = Dense(units_layer4, activation='relu')(x)
    out = Dense(20, activation=None)(x)
    model = Model(inputs=inp, outputs=out)

    if optimizer == 'sgd':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=0.001)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
    return model


# def inverse_network(units_layer1=64, units_layer2=512, units_layer3=1024, units_layer4=2048, units_layer5=2048, units_layer6=256, optimizer='rmsprop'):
#     inp = Input(shape=(20,), name='forward_input')
#     x = inp
#     x = Dense(units_layer1)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer2)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer3)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer4)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer5)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(units_layer6)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     out = Dense(9, activation=None)(x)
#     model = Model(inputs=inp, outputs=out)
#
#     if optimizer == 'sgd':
#         optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
#     elif optimizer == 'adam':
#         optimizer = Adam(learning_rate=0.001)
#     elif optimizer == 'rmsprop':
#         optimizer = RMSprop(learning_rate=0.001)
#
#     model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
#     return model



def inverse_network(units_layer1=64, units_layer2=512, units_layer3=1024, units_layer4=2048, units_layer5=2048, units_layer6=256, optimizer='rmsprop'):
    inp = Input(shape=(20,), name='forward_input')
    x = inp
    x = Dense(units_layer1, activation='relu')(x)
    x = residual_block(x, units_layer1)
    x = Dense(units_layer2, activation='relu')(x)
    x = residual_block(x, units_layer2)
    x = Dense(units_layer3, activation='relu')(x)
    x = residual_block(x, units_layer3)
    x = Dense(units_layer4, activation='relu')(x)
    x = residual_block(x, units_layer4)
    x = Dense(units_layer5, activation='relu')(x)
    x = residual_block(x, units_layer5)
    x = Dense(units_layer6, activation='relu')(x)

    out = Dense(4, activation=None)(x)
    model = Model(inputs=inp, outputs=out)

    if optimizer == 'sgd':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=0.001)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
    return model

# def dense_block(x, num_layers, growth_rate):
#     for _ in range(num_layers):
#         cb = composite_function(x, growth_rate)
#         x = Concatenate()([x, cb])
#     return x
#
# def composite_function(x, units):
#     x = Dense(units, activation='relu')(x)
#     return x
#
#
# def inverse_network(input_shape, output_units, num_dense_blocks, num_layers_per_block, growth_rates, optimizer='rmsprop'):
#     inputs = Input(shape=input_shape)
#     x = Dense(64, activation='relu')(inputs)
#
#     for i in range(num_dense_blocks):
#         growth_rate = growth_rates[i]
#         x = dense_block(x, num_layers=num_layers_per_block, growth_rate=growth_rate)
#
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     outputs = Dense(output_units, activation=None)(x)
#
#     model = Model(inputs, outputs)
#     if optimizer == 'sgd':
#         optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
#     elif optimizer == 'adam':
#         optimizer = Adam(learning_rate=0.001)
#     elif optimizer == 'rmsprop':
#         optimizer = RMSprop(learning_rate=0.001)
#
#     model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
#     return model



