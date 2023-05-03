import tensorflow as tf
import numpy as np
import os

def AlexNet(args, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
    # define AlexNet model
    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE, embeddings_initializer=tf.keras.initializers.HeNormal(),
                                          input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
    x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
    x = tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_1')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
#    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
#    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#    x = tf.keras.layers.Activation('relu')(x)
#    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=tf.keras.initializers.HeNormal(), name='last_dense')(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    model = tf.keras.models.Model(read_input, output, name='AlexNet')

    # get types of the layers
    #print(f'embedding layer: {embedding}')
#    print(f'dense1 layer info: {dense1}')
#    print(f'dtype of dense1: {dense1.dtype_policy}')
#    print(f'x.dtype: {x.dtype.name}')
#    print(f'dense1.kernel.dtype: {dense1.kernel.dtype.name}')
#    print('Outputs dtype: %s' % output.dtype.name)
#    for idx in range(len(model.layers)):
#        print(f'INDEX: {idx} - NAME: {model.get_layer(index = idx).name} - TYPE: {model.get_layer(index = idx).dtype}')

    with open(os.path.join(args.output_dir, f'model-alexnet.txt'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

def VGG16(output_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
    # define AlexNet model
    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE,
                                          input_length=VECTOR_SIZE, mask_zero=True, trainable=True)(x)
    x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096)(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=4096)(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=1000)(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(NUM_CLASSES)(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    model = tf.keras.models.Model(read_input, output, name='VGG16')

    with open(os.path.join(output_dir, 'model-vgg16.txt'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model


def VDCNN(output_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE):

    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE,
                                          input_length=VECTOR_SIZE, mask_zero=True, trainable=True)(x)
    x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    # first convolutional block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
    # second convolutional block
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
    # third convolutional block
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
    # fourth convolutional block
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Apply k-max pooling with k=8 to extract the k most important features independently of the position they appear in the read
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    # first fully connected layer
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # second fully connect layer
    x = tf.keras.layers.Dense(units=2048)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # third fully connected layer
    x = tf.keras.layers.Dense(NUM_CLASSES)(x)
    output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    model = tf.keras.models.Model(read_input, output, name='VDCNN')

    with open(os.path.join(output_dir, 'model-vdcnn.txt'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
