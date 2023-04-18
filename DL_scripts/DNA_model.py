import tensorflow as tf
import numpy as np
import os

def DNA_net(args, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
    # define AlexNet model
    n_rows = args.n_rows
    n_cols = args.n_cols
    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    # read_input = tf.keras.layers.Input(shape=(n_rows, n_cols), dtype='int32')
    x = read_input
    print(x.shape)
    print(x)
    x = tf.keras.layers.Reshape((n_rows, n_cols))(x)
    # x = tf.reshape(x, [None, n_rows, n_cols])
    print(x.shape)
    print(x)
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE, embeddings_initializer=tf.keras.initializers.HeNormal(),
                                          input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
    print(x.shape)
    print(x)
    x = tf.keras.layers.Reshape((n_rows, n_cols*EMBEDDING_SIZE, 1))(x)  # output shape: (n_rows, n_cols*EMBEDDING_SIZE, 1)
    print(x.shape)
    x = tf.keras.layers.Conv2D(96, kernel_size=(args.kernel_height, 3*EMBEDDING_SIZE), strides=(1, EMBEDDING_SIZE), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_1')(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(args.kernel_height, 4), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    print(x.shape)
    x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    print(x.shape)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    print(x.shape)
    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
#    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
#    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#    x = tf.keras.layers.Activation('relu')(x)
#    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=tf.keras.initializers.HeNormal(), name='last_dense')(x)
    print(x.shape)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    model = tf.keras.models.Model(read_input, output, name='AlexNet')

    if args.output_dir is True:
        with open(os.path.join(args.output_dir, f'dna-model.txt'), 'w+') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model