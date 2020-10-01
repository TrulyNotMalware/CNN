import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y)= fashion_mnist.load_data()

train_X = train_X /255.0
test_X = test_X / 255.0
plt.imshow(train_X[0])
plt.colorbar()
plt.show()

print(train_X.shape)
print(test_X.shape)

train_X = train_X.reshape(-1,28,28,1)
test_X = test_X.reshape(-1,28,28,1)

input = layers.Input(shape=(28,28,1))
t = layers.Conv2D(filters=16,kernel_size=(5,5),strides=1)(input)
t = layers.BatchNormalization(axis=1)(t)
t = layers.Activation('relu')(t)
t = layers.MaxPooling2D(pool_size=2,strides=1)(t)
t = layers.Dropout(0.2)(t)
#f1 = layers.ZeroPadding2D(padding=(1))(t)
f1 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=1,padding='SAME')(t)
f1 = layers.BatchNormalization(axis=1)(f1)
t = layers.Add()([f1,t])
t = layers.Activation('relu')(t)
t = layers.MaxPooling2D(pool_size=2,strides=1)(t)
t = layers.Dropout(0.2)(t)

t = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='SAME')(t)
t = layers.BatchNormalization(axis=1)(t)

f2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='SAME')(t)
f2 = layers.BatchNormalization(axis=1)(f2)

f3 = layers.Conv2D(filters=32,kernel_size=(2,2),strides=1,padding='SAME')(t)
f3 = layers.BatchNormalization(axis=1)(f3)
t = layers.Add()([f2,f3,t])
t = layers.Activation('relu')(t)

t = layers.MaxPooling2D(pool_size=2,strides=1)(t)
t = layers.GlobalAveragePooling2D()(t)
t = layers.Dense(units=32,activation='relu')(t)
t = layers.Dense(units=10,activation='softmax')(t)


model = tf.keras.Sequential([
    layers.Conv2D(filters=16,kernel_size=(3,3),strides=1,input_shape=(28,28,1)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=2,strides=1),
    layers.Dropout(0.2),
    layers.Conv2D(filters=32,kernel_size=(2,2),strides=1),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=2,strides=1),
    layers.GlobalAveragePooling2D(),
    layers.Dense(units=64,activation='relu'),
    layers.Dense(units=32,activation='relu'),
    layers.Dense(units=10,activation='softmax')
])
model2 = tf.keras.Sequential([
    layers.Conv2D(filters=16,kernel_size=(3,3),input_shape=(28,28,1)),
    layers.Conv2D(filters=32,kernel_size=(3,3)),
    layers.Conv2D(filters=64,kernel_size=(3,3)),
    layers.Flatten(),
    layers.Dense(units=128,activation='relu'),
    layers.Dense(units=10,activation='softmax')
])
'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),input_shape=(28,28,1),strides=1))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=1))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),strides=1))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=1))
model.add(layers.Flatten())
model.add(layers.Dense(units=32,activation='relu'))
model.add(layers.Dense(units=10,activation='softmax'))
'''
'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=1,input_shape=(28,28,1),kernel_initializer='he_normal'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=1))
model.add(layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1,kernel_initializer='he_normal'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=64,activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
'''
model = tf.keras.models.Model(inputs=[input],outputs=t)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(train_X,train_Y,epochs=30,validation_split=0.25)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],'b-',label='loss')
plt.plot(history.history['val_loss'],'r--',label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'g-',label='accuracy')
plt.plot(history.history['val_accuracy'],'k--',label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()
plt.show()

#model.evaluate(test_X,test_Y,verbose=0)
