import tensorflow as tf
import os


def VGG16(args, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):

    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    if args.DNA_model:
        x = tf.keras.layers.Reshape((args.n_rows, args.n_cols))(x)
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE + 1, output_dim=EMBEDDING_SIZE,
                                  embeddings_initializer=tf.keras.initializers.HeNormal(),
                                  input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
    if args.DNA_model:
        x = tf.keras.layers.Reshape((args.n_rows, args.n_cols * EMBEDDING_SIZE, 1))(x)  # output shape: (n_rows, n_cols*EMBEDDING_SIZE, 1)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_1')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_2')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_3')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_4')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_5')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_6')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_7')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_8')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_9')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_10')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_11')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_12')(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', name='conv_13')(x)
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

    with open(os.path.join(args.output_dir, 'model-vgg16.txt'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model