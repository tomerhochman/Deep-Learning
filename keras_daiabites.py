import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\1\OneDrive\Desktop\Projects\Keras introduction\diabetes.csv",sep=',')

X = df.values[:, 0:8]
Y = df.values[:, 8]


model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.30, epochs=150, batch_size=10, verbose=1)
Loss, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: ', Loss)

predictions = model.predict(X)
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')