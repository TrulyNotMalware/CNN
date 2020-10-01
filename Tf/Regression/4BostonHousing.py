import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import boston_housing
(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

print(train_X[0])
print(train_Y[0])

#데이터릂 정규화 한다. 평균=0, 표편=1
# (데이터값 - 평균)/표준편차
x_mean = train_X.mean(axis=0)
y_mean = train_Y.mean(axis=0)
print("train_x's mean :",x_mean)
print("train_y's mean :",y_mean)
x_std = train_X.std(axis=0)
y_std = train_Y.std(axis=0)
print("train_x's std :",x_std)
print("train_x's std :",y_std)

train_X -= x_mean
train_X /= x_std
train_Y -= y_mean
train_Y /= y_std

print(train_X[0])
print(train_Y[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
#y= 주택가격 하나. node또한 한개
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
model.summary()
#validation_split -> 25%만큼 훈련데이터에서 떼서 검증데이터로 씀.
history = model.fit(train_X,train_Y,epochs=25,batch_size=32,validation_split=0.25,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss')])
#callback은 patience만큼 monitor변수가 더 이상 최적의 값을 넘어서지 못하면, 학습을 EarlyStopping.
#test_X, test_Y 정규화
tx_mean = test_X.mean(axis=0)
ty_mean = test_Y.mean(axis=0)
tx_std = test_X.std(axis=0)
ty_std = test_Y.std(axis=0)
test_X -= tx_mean
test_X /= tx_std
test_Y -= ty_mean
test_Y /= ty_std

plt.plot(history.history['loss'],'b-', label='loss')
plt.plot(history.history['val_loss'],'r--',label='val_loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

model.evaluate(test_X, test_Y)

pred_Y = model.predict(test_X)
print(pred_Y)

plt.figure(figsize=(5,5))
plt.plot(test_Y,pred_Y,'b.')
plt.axis([min(test_Y),max(test_Y),min(test_Y),max(test_Y)])
plt.plot([min(test_Y),max(test_Y)],[min(test_Y),max(test_Y)],ls="--", c='.3',label='y=x')
plt.xlabel('test_Y')
plt.ylabel('pred_Y')
plt.show()
