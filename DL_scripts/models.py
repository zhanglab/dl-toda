import torch
import torch.nn as nn
import os

class AlexNet(nn.Module):
    def __init__(self, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
        super(AlexNet, self).__init__()
        # self.VECTOR_SIZE = VECTOR_SIZE
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE+1,
            embedding_dim=EMBEDDING_SIZE,
            padding_idx=0 # padding do not contribute to the gradient
        )
        nn.init.kaiming_normal_(self.embedding.weight)
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=5),
            nn.BatchNorm2d(96, eps=1e-2, momentum=0.01), # momentum: TF=0.99 (1 - PyTorch momentum)
            nn.ReLU(),
            
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(256, eps=1e-2, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(384, eps=1e-2, momentum=0.01),
            nn.ReLU(),
            
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(384, eps=1e-2, momentum=0.01),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256, eps=1e-2, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Dynamically calculate the flattened feature size
        self._flattened_size = self._get_conv_output((1, VECTOR_SIZE, EMBEDDING_SIZE))
        
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, 4096),
            nn.BatchNorm1d(4096, eps=1e-2, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000, eps=1e-2, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1000, NUM_CLASSES),
            nn.BatchNorm1d(NUM_CLASSES, eps=1e-2, momentum=0.01),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights using HeNormal (kaiming_normal_ in PyTorch)
        self._initialize_weights()

    def _get_conv_output(self, shape):
        # Helper to calculate the exact dimensions after conv/pooling operations
        batch_size = 1
        input_dummy = torch.rand(batch_size, *shape)
        output_dummy = self.features(input_dummy)
        return output_dummy.data.view(batch_size, -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape expected: (batch_size, sequence_length)
        x = self.embedding(x)
        
        # Reshape to (batch_size, channels=1, height, width) for 2D Convolutions
        # PyTorch Conv2d expects channel-first format (N, C, H, W)
        x = x.unsqueeze(1) 
        
        x = self.features(x)
        
        # Flatten for Dense layers
        x = x.view(x.size(0), -1) 
        
        output = self.classifier(x)
        return output



# import tensorflow as tf
# import numpy as np
# import os

# def AlexNet(args, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE, output_dir=False):
#     # define AlexNet model
#     read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
#     x = read_input
#     x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE, embeddings_initializer=tf.keras.initializers.HeNormal(),
#                                           input_length=VECTOR_SIZE, mask_zero=True, trainable=True, name='embedding')(x)
#     x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
#     x = tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), name='conv_1')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
# #    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
# #    x = tf.keras.layers.Dense(units=4096, kernel_initializer=tf.keras.initializers.HeNormal())(x)
# #    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
# #    x = tf.keras.layers.Activation('relu')(x)
# #    x = tf.keras.layers.Dropout(0.5)(x)
#     x = tf.keras.layers.Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal())(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
#     x = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=tf.keras.initializers.HeNormal(), name='last_dense')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99)(x)
#     output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
#     model = tf.keras.models.Model(read_input, output, name='AlexNet')

#     # get types of the layers
#     #print(f'embedding layer: {embedding}')
# #    print(f'dense1 layer info: {dense1}')
# #    print(f'dtype of dense1: {dense1.dtype_policy}')
# #    print(f'x.dtype: {x.dtype.name}')
# #    print(f'dense1.kernel.dtype: {dense1.kernel.dtype.name}')
# #    print('Outputs dtype: %s' % output.dtype.name)
# #    for idx in range(len(model.layers)):
# #        print(f'INDEX: {idx} - NAME: {model.get_layer(index = idx).name} - TYPE: {model.get_layer(index = idx).dtype}')
#     if output_dir is True:
#         with open(os.path.join(args.output_dir, f'model-alexnet.txt'), 'w+') as f:
#             model.summary(print_fn=lambda x: f.write(x + '\n'))

#     return model

# def VGG16(output_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE, DROPOUT_RATE):
#     # define AlexNet model
#     read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
#     x = read_input
#     x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE,
#                                           input_length=VECTOR_SIZE, mask_zero=True, trainable=True)(x)
#     x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#     x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(units=4096)(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dense(units=4096)(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dense(units=1000)(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dense(NUM_CLASSES)(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
#     model = tf.keras.models.Model(read_input, output, name='VGG16')
#
#     with open(os.path.join(output_dir, 'model-vgg16.txt'), 'w+') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#
#     return model
#
#
# def VDCNN(output_dir, VECTOR_SIZE, EMBEDDING_SIZE, NUM_CLASSES, VOCAB_SIZE):
#
#     read_input = tf.keras.layers.Input(shape=(VECTOR_SIZE), dtype='int32')
#     x = read_input
#     x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_SIZE,
#                                           input_length=VECTOR_SIZE, mask_zero=True, trainable=True)(x)
#     x = tf.keras.layers.Reshape((VECTOR_SIZE, EMBEDDING_SIZE, 1))(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     # first convolutional block
#     x = tf.keras.layers.Conv2D(64, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
#     # second convolutional block
#     x = tf.keras.layers.Conv2D(128, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(128, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
#     # third convolutional block
#     x = tf.keras.layers.Conv2D(256, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
#     # fourth convolutional block
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(512, kernel_size=(3, EMBEDDING_SIZE), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     # Apply k-max pooling with k=8 to extract the k most important features independently of the position they appear in the read
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
#     # first fully connected layer
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(units=4096)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     # second fully connect layer
#     x = tf.keras.layers.Dense(units=2048)(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     # third fully connected layer
#     x = tf.keras.layers.Dense(NUM_CLASSES)(x)
#     output = tf.keras.layers.Activation('softmax', dtype='float32',)(x)
#     model = tf.keras.models.Model(read_input, output, name='VDCNN')
#
#     with open(os.path.join(output_dir, 'model-vdcnn.txt'), 'w+') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#
#     return model
