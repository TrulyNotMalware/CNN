import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
])
#lr = learning rate
#SGD = Stochastic Gradient Descent
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),loss='mse')

model.summary()
log = model.fit(x,y,epochs=2000, batch_size=1)
predict = model.predict(x)
print(predict)

plt.plot(log.history['loss'])
plt.show()