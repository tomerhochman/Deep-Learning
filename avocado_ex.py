import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\1\OneDrive\Desktop\Pycharm_projects\Tensorflow_less\avocado.csv", sep=',')

#################################-----PREPROCESS THE DATA----################################################

# Drop unused columns
df.drop(['Unnamed: 0', 'region'], axis=1, inplace=True)

# Change Date to month and day
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].map(lambda x: x.month)
df['Day'] = df['Date'].map(lambda x: x.day)

# Since we have a year,month,day column -we can drop the date column
df.drop('Date', axis=1, inplace=True)

# Change the 'type column to binary ( 0 or 1)
df['type'].replace(['conventional', 'organic'], [0, 1], inplace=True)

#############################------MODEL STRUCTURE--------#################################################

X = df.iloc[:, 1:14]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = tf.constant(X_train.astype('float32'))
y_train = tf.constant(y_train.astype('float32'))
X_test = tf.constant(X_test.astype('float32'))
y_test = tf.constant(y_test.astype('float32'))

w1 = tf.Variable(tf.random.normal((12, 15), mean=0.0), name='w1')
b1 = tf.Variable(tf.zeros(15, dtype=tf.float32), name='b1')
w2 = tf.Variable(tf.random.normal((15, 18), mean=0.0), name='w2')
b2 = tf.Variable(tf.zeros(18, dtype=tf.float32), name='b2')
w3 = tf.Variable(tf.random.normal((18, 1), mean=0.0), name='w2')
b3 = tf.Variable(tf.zeros(1, dtype=tf.float32), name='b2')

# TODO: figure out the right buffer size and batch size ratio
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# val_dataset = val_dataset.batch(batch_size)

loss_lst = []
epochs = 5
lr = 0.01

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    # TODO: solve the fluctuations problem

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:
            l1 = tf.nn.relu(x_batch_train @ w1 + b1)
            l2 = tf.nn.relu(l1 @ w2 + b2)
            yp = tf.nn.sigmoid(l2 @ w3 + b3)
            loss = tf.reduce_mean((yp - y_batch_train) ** 2)
            loss_lst.append((loss.numpy()))

        [dw3, db3] = tape.gradient(loss, [w3, b3])
        [dw2, db2] = tape.gradient(loss, [w2, b2])
        [dw1, db1] = tape.gradient(loss, [w1, b1])

        w3.assign_sub((lr * dw3))
        b3.assign_sub((lr * db3))
        w2.assign_sub((lr * dw2))
        b2.assign_sub((lr * db2))
        w1.assign_sub((lr * dw1))
        b1.assign_sub((lr * db1))

        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss)))
            print("Seen so far: %s samples" % ((step + 1) * 500))

print('loss: ', loss.numpy())
plt.plot(loss_lst)
plt.grid()
