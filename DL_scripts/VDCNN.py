import tensorflow as tf
import os


def VDCNN(output_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE):
    n_rows = args.n_rows
    n_cols = args.n_cols
    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    x = tf.keras.layers.Reshape((n_rows, n_cols))(x)
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE + 1, output_dim=EMBEDDING_SIZE,
                                  embeddings_initializer=tf.keras.initializers.HeNormal(),
                                  input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
    x = tf.keras.layers.Reshape((n_rows, n_cols * EMBEDDING_SIZE, 1))(x)  # output shape: (n_rows, n_cols*EMBEDDING_SIZE, 1)
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
