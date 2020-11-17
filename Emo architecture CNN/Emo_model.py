import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

### Create training and validation sets from FER+ dataset
# state directories with images
dirtrain = '/home/becode/AI/Data/Bagaar/Fer2013/fer2013Plus/FER2013Train'
dirval = '/home/becode/AI/Data/Bagaar/Fer2013/fer2013Plus/FER2013TestValid'
# reading in csv with image name and highest emotion label
train = pd.read_csv('/home/becode/AI/Data/Bagaar/Fer2013/fer2013Plus/FER2013Train/train1.csv',usecols=['Image name','emo'])
vali = pd.read_csv('/home/becode/AI/Data/Bagaar/Fer2013/fer2013Plus/FER2013TestValid/vali1.csv', usecols=['Image name','emo'])
train['emo'] = train['emo'].astype(str)
vali['emo'] = vali['emo'].astype(str)

### Using Imagedatagenerator.flow_from_dataframe() to create subsets
# x_col contains the relative filepath to directory, y_col contains the 8 emotion labels
# Keras interprets this so it generates the categorical 8 output classes
train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=40,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_dataframe(
    train,
    directory=dirtrain,
    x_col="Image name",
    y_col='emo',
    weight_col=None,
    target_size=(64, 64),
    color_mode="grayscale",
    classes= None,#['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown'],
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=123,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
    interpolation="nearest",
    validate_filenames=True,
)
val_ds = val_datagen.flow_from_dataframe(
   vali,
    directory=dirval,
    x_col="Image name",
    y_col='emo',
    weight_col=None,
    target_size=(64, 64),
    color_mode="grayscale",
    classes= None,#['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown'],
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=123,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
    interpolation="nearest",
    validate_filenames=True,
)

### CNN model
# Set params
batch_size = 64
epochs = 50 # 10
# num_classes = 1 # O or 1

# CNN Model architecture
model = Sequential()
model.add(Conv2D(input_shape=(64,64,1), filters=32, kernel_size=(4,4), activation='relu', data_format='channels_last',
                 kernel_regularizer='l2'))
model.add(Conv2D(32, kernel_size=(4,4),activation='relu', data_format='channels_last', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(4,4), activation='relu', data_format='channels_last', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(4,4), activation='relu', data_format='channels_last', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(4,4), activation='relu', data_format='channels_last', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(4,4), activation='relu', data_format='channels_last', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(8, activation="softmax")) # softmax for multiple output

# Compile model
model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
reduce_lr = keras.callbacks.ReduceLROnPlateau()
#early = keras.callbacks.EarlyStopping(patience=3)
modelleke = model.fit(train_ds,batch_size=batch_size, epochs=epochs, verbose=1,validation_data=val_ds,
               callbacks=[reduce_lr])

# Save model and weights
model.save("model_emopy_50_ep.h5")

### Model Evaluation on validation set
# Evaluate Test set
test_eval = model.evaluate(val_ds, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Plot accuracy and loss plots
accuracy = modelleke.history['accuracy']
val_accuracy = modelleke.history['val_accuracy']
loss = modelleke.history['loss']
val_loss = modelleke.history['val_loss']
epochs = range(len(accuracy))
plt.figure()
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Train-Val Acc ep50')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Train-Val Loss ep50')
#plt.show()