import tensorflow as tf
import os

def DNA_net(args, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
    # define AlexNet model
    n_rows = args.n_rows
    n_cols = args.n_cols
    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    x = tf.keras.layers.Reshape((n_rows, n_cols))(x)
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE, embeddings_initializer=tf.keras.initializers.HeNormal(),
                                          input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
    x = tf.keras.layers.Reshape((n_rows, n_cols*EMBEDDING_SIZE, 1))(x)  # output shape: (n_rows, n_cols*EMBEDDING_SIZE, 1)
    x = tf.keras.layers.Conv2D(96, kernel_size=(args.kh_conv_1, args.kw_conv_1*EMBEDDING_SIZE), strides=(args.sh_conv_1, EMBEDDING_SIZE), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_1')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(args.kh_conv_2, args.kw_conv_2), strides=(args.sh_conv_2, args.sw_conv_2), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_2')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_3')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_4')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_5')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=tf.keras.initializers.HeNormal(), name='last_dense')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    model = tf.keras.models.Model(read_input, output, name='DNANet')

    with open(os.path.join(args.output_dir, f'dna-model.txt'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model