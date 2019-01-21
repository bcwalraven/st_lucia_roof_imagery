
# coding: utf-8

# In[9]:


import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[2]:


# Initilzing the CNN
cnn = Sequential()

# Step 1 - create the convolution layer
cnn.add(Convolution2D(filters = 32, 
                      kernel_size = (5, 5), 
                      activation = 'relu', 
                      input_shape = (150, 150, 3)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Convolution2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Convolution2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

 # Step 3 - Flatten to create the input vector
cnn.add(Flatten())

# Step 4 - add the fully connected layer
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units = 3, activation = 'softmax'))


# In[3]:


# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[4]:


# Part 2 - fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split = .2)

test_datagen = ImageDataGenerator(rescale=1./255,
                                  validation_split = .2)


# In[17]:


training_set = train_datagen.flow_from_directory('./re-sorted_roof_images/',
                                                 target_size=(150, 150),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 subset='training')

validation_set = test_datagen.flow_from_directory('./re-sorted_roof_images/',
                                                   target_size=(150, 150),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle = False,
                                                   subset='validation')


# In[8]:


cnn.fit_generator(training_set,
                        steps_per_epoch=len(training_set),
                        epochs=100,
                        validation_data=validation_set,
                        validation_steps = len(validation_set),
                        callbacks = [EarlyStopping(monitor='val_loss', 
                                     patience=30, 
                                     mode='auto', 
                                     restore_best_weights=True)]
                        )


# In[ ]:


#cnn.save('./pickled_models/categorical_model.h5')


# In[18]:


y = validation_set.classes
y_pred = np.argmax(cnn.predict_generator(validation_set, len(validation_set)), axis=1)


# In[21]:


print('Model Accuracy:')
print(round(accuracy_score(y, y_pred), 4))
print()
print('Confusion Matrix:')
print(confusion_matrix(validation_set.classes, y_pred))
print()
target_names = ['None', 'Good', 'Poor']
print('Classification Report:')
print(classification_report(validation_set.classes, y_pred, target_names=target_names))

