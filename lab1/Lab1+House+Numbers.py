
# coding: utf-8

# In[ ]:

import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import scipy.io
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D


# ### Load Data

# In[ ]:

NUM_TRAIN_SAMPLES = 73257
NUM_TEST_SAMPLES = 26032
IMAGE_SIZE = 32
RGB = 3


# In[ ]:

train = scipy.io.loadmat('data/mat/train_32x32.mat')
X_train = np.asarray([train['X'][:,:,:,i] for i in range(NUM_TRAIN_SAMPLES)])
y_train = train['y']


# In[ ]:

print ('X train: {0} y train: {1}'.format(X_train.shape, y_train.shape))


# In[ ]:

test = scipy.io.loadmat('data/mat/test_32x32.mat')
X_test = np.asarray([test['X'][:,:,:,i] for i in range(NUM_TEST_SAMPLES)])
y_test = test['y']


# In[ ]:

print ('X test: {0} y test: {1}'.format(X_test.shape, y_test.shape))


# In[ ]:

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0


# In[ ]:

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()


# ### Define Architecture
# ### 3-Layer CNN with 3x3 filter size and 32, 64, and 128 filters.
# ### Alternating 2x2 average pooling layers between each convolution
# ### ReLu activation
# ### One softmax and output layer
# ### Adam loss function to train

# In[ ]:

model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        AveragePooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        AveragePooling2D((2, 2))
    ])


# In[ ]:

model.summary()


# In[ ]:

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(11))


# In[ ]:

model.summary()


# In[ ]:

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ### Train Model for 10 epochs

# In[ ]:

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))


# ### Evaluate Model - 90% accuracy

# In[ ]:

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)


# In[ ]:

test_acc


# ### Review

# 0. Normalized images during pre-processing
# 1. Experimented with varying number of layers and found that a three-layer convolutional network alternating pooling layers provided the best results
# 2. Experimented with 32, 64, and 128 filter sizes to extract more detail from each layer
# 3. Applied a 3x3 filter size for each convolutional layer
# 4. Used an average pooling layer between each convolution because contrast between house numbers and backgrounds were not clearly defined
# 4. Used ReLu activation function to quickly train the model
# 5. Achieved 90% accuracy

# In[ ]:



