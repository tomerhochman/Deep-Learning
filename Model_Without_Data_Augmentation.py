import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.losses import CategoricalCrossentropy

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import cv2
import os
import shutil
import pandas as pd
import numpy as np

########################  Data Preprocessing  #######################################
'''
Creating train and test folders.
splitting data to train and test.

our data structure:

  - Images_Input parent folder (parent)
    - Train (child)
       - covid (sub)
       - normal (sub) 
     - valid (child)
       - covid (sub)
       - normal (sub)  

'''

# Home directory
home_path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Model_Without_Augmentation\Images_Input'

# Create train and validation directories
train_path = os.path.join(home_path, 'train')
os.mkdir(train_path)
val_path = os.path.join(home_path, 'valid')
os.mkdir(val_path)

# Create sub-directories
covid_train_path = os.path.join(home_path + r'/train', 'covid')
os.mkdir(covid_train_path)

normal_train_path = os.path.join(home_path + r'/train', 'normal')
os.mkdir(normal_train_path)

covid_val_path = os.path.join(home_path + r'/valid', 'covid')
os.mkdir(covid_val_path)

normal_val_path = os.path.join(home_path + r'/valid', 'normal')
os.mkdir(normal_val_path)

# Original df
df = pd.read_csv(r'Model_Without_Augmentation/images.csv')

# Images and Labels
X = df.loc[:, 'Image_Name']
y = df.loc[:, 'Class']

# Train-Test split for train and validation images
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=27, stratify=y)

# Train df
df_train = pd.DataFrame(columns=['Image_Name', 'Class'])
df_train['Image_Name'] = train_x
df_train['Class'] = train_y

# Validation df
df_valid = pd.DataFrame(columns=['Image_Name', 'Class'])
df_valid['Image_Name'] = val_x
df_valid['Class'] = val_y

# Reset index
df_train.reset_index(drop=True, inplace=True)
df_valid.reset_index(drop=True, inplace=True)

path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Model_Without_Augmentation'
normal_train_path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Model_Without_Augmentation\Images_Input\train\normal'
covid_train_path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Model_Without_Augmentation\Images_Input\train\covid'
normal_val_path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Model_Without_Augmentation\Images_Input\valid\normal'
covid_val_path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Model_Without_Augmentation\Images_Input\valid\covid'

# Save train images
for i in range(len(df_train)):

    image = df_train.loc[i, 'Image_Name']

    if df_train.loc[i, 'Class'] == 0:
        shutil.copy(path + r'/Images_Input/' + image, normal_train_path)
    else:
        shutil.copy(path + r'/Images_Input/' + image, covid_train_path)

# Save validation images
for i in range(len(df_valid)):

    image = df_valid.loc[i, 'Image_Name']

    if df_valid.loc[i, 'Class'] == 0:
        shutil.copy(path + r'/Images_Input/' + image, normal_val_path)
    else:
        shutil.copy(path + r'/Images_Input/' + image, covid_val_path)

####################### Visualize the data #######################################

# Plot images count with seaborn

l = []
for i in df_train.iloc[:, 1]:
    if i == 0:
        l.append("normal")
    else:
        l.append("covid")

sns.set_style('darkgrid')
sns.countplot(l)

######################## Building a Model with Keras without Data Augmentation  ######################

# Images
train_images = df_train.loc[:, 'Image_Name']
train_labels = df_train.loc[:, 'Class']

test_images = df_valid.loc[:, 'Image_Name']
test_labels = df_valid.loc[:, 'Class']

# Train images
x_train = []
for i in train_images:
    image = home_path + '/' + i
    img = cv2.imread(image)
    x_train.append(img)

# Train labels
y_train = to_categorical(train_labels)

# Test images
x_test = []
for i in test_images:
    image = home_path + '/' + i
    img = cv2.imread(image)
    x_test.append(img)

# Test labels
y_test = to_categorical(test_labels)

# Normalize images
x_train = np.array(x_train, dtype="float") / 255.0
x_test = np.array(x_test, dtype="float") / 255.0

# Model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.24))
model.add(Dense(2, activation='softmax'))

with open('Model_Summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Compile
cce = CategoricalCrossentropy(from_logits=False)
optim = Adam()
model.compile(loss=cce, optimizer=optim, metrics=['accuracy'])

# Fit
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32, verbose=1)

# Plot results
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Val Accuracy'], loc='upper left')
plt.show()

# Model evaluation
model.evaluate(x_test, y_test, verbose=1)

# Model prediction
predictions = model.predict(x=x_test, verbose=1)
predictions = np.round(predictions)

# Model Analyzing
cm = confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=predictions.argmax(axis=1))
disp = ConfusionMatrixDisplay(cm, display_labels=['covid', 'normal'])
disp.plot()

print(classification_report(y_true=y_test.argmax(axis=1), y_pred=predictions.argmax(axis=1)))



