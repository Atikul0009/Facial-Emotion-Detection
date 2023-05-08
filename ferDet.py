import numpy as np
from PIL import Image
import pandas as pd
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import cv2
import scikitplot
import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# Define the image size
img_size = 256

# Load the FER 2013 dataset
data = pd.read_csv('/root/DataPrep/new/fer2013_updated.csv')

# Split the dataset into training and validation sets
train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']


# Convert the emotion column to string type
train_data['emotion'] = train_data['emotion'].astype(str)
val_data['emotion'] = val_data['emotion'].astype(str)






# Define a data generator for image loading and preprocessing
def preprocess_input(x):
    x = x.astype('float32')
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def load_image(image_path):
    img = load_img(image_path, target_size=(img_size, img_size), interpolation='cubic')
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Define a data generator for loading and augmenting the images
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='pixels',
    y_col='emotion',
    target_size=(img_size, img_size),
    batch_size=64,
    class_mode='categorical',
    interpolation='nearest',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    x_col='pixels',
    y_col='emotion',
    target_size=(img_size, img_size),
    batch_size=64,
    class_mode='categorical',
    interpolation='nearest',
    shuffle=False
)


# Check the number of samples in the training and validation sets
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", val_generator.samples)




# Define the ResNet50 model as a feature extractor
base_model = tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Add a global average pooling layer and a fully connected layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers except the last two layers
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Compile the model with a categorical cross-entropy loss function and Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(train_generator,
          validation_data=val_generator,
          steps_per_epoch=len(train_generator),
          epochs=25,
          callbacks=[early_stopping])

# Save the trained model
model.save('fer2013_Nasnet_256_nearest.h5')


sns.set()
fig = pyplot.figure(0, (12, 4))

ax = pyplot.subplot(1, 2, 1)
sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='train')
sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='valid')
pyplot.title('Accuracy')
pyplot.tight_layout()

ax = pyplot.subplot(1, 2, 2)
sns.lineplot(x=history.epoch, y=history.history['loss'], label='train')
sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='valid')
pyplot.title('Loss')
pyplot.tight_layout()

pyplot.savefig('epoch_history_Nasnet_256_nearest.png')




model=tf.keras.models.load_model('/root/DataPrep/new/fer2013_Nasnet_256_nearest.h5')



mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}
label_names = list(mapper.values())
print(label_names)

yhat_valid = np.argmax(model.predict(val_generator), axis=1)

# true_label = np.argmax(y_valid, axis=1)

true_label = val_generator.classes

pred_value = [mapper[i] for i in yhat_valid]

true_value = [mapper[i] for i in true_label]


title  = 'Confusion Matrix NasNetLarge'
scikitplot.metrics.plot_confusion_matrix(true_value, pred_value,labels= label_names,title=title, figsize=(7,7),x_tick_rotation=15)


pyplot.savefig("confusion_matrix_nasnet.png",dpi=300,bbox_inches='tight')

print(classification_report(true_value, pred_value,labels= label_names ))


