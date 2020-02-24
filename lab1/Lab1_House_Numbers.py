
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

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


# ### Load matrices

# In[ ]:

NUM_TRAIN_SAMPLES = 73257
NUM_TEST_SAMPLES = 26032
IMAGE_SIZE = 32
RGB = 3


# In[ ]:

train = scipy.io.loadmat('train_32x32.mat')
X_train = np.asarray([train['X'][:,:,:,i] for i in range(NUM_TRAIN_SAMPLES)])
y_train = train['y']


# In[ ]:

print ('X train: {0} y train: {1}'.format(X_train.shape, y_train.shape))


# In[ ]:

test = scipy.io.loadmat('test_32x32.mat')
X_test = np.asarray([test['X'][:,:,:,i] for i in range(NUM_TEST_SAMPLES)])
y_test = test['y']


# In[ ]:

print ('X test: {0} y test: {1}'.format(X_test.shape, y_test.shape))


# In[ ]:

#X_train, X_test = X_train / 255.0, X_test / 255.0


# In[ ]:

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()


# ### Train model

# In[ ]:

model = tf.keras.Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2))
    ])


# In[ ]:

model.summary()


# In[ ]:

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(11))


# In[ ]:

model.summary()


# In[ ]:

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))


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


# In[ ]:



