import tensorflow as tf
import os

def LSTM(args, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
    # define AlexNet model
    read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
    x = read_input
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE, embeddings_initializer=tf.keras.initializers.HeNormal(),
                                          input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
#    x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=tf.keras.initializers.HeNormal(), name='last_dense')(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
    output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
    model = tf.keras.models.Model(read_input, output, name='LSTM')

    with open(os.path.join(args.output_dir, f'model-lstm.txt'), 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model