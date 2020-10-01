import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rand

x = np.array(tf.random.normal([20],0,1))
y = np.array(tf.random.uniform([20],11,14))

a= tf.Variable(rand.random())
b= tf.Variable(rand.random())
c= tf.Variable(rand.random())

def compute_loss():
    y_pred= a*x*x + b*x + c #NonLinear
    loss = tf.reduce_mean((y-y_pred)**2)
    return loss

optimizer = tf.keras.optimizers.Adam(lr=0.07)

for i in range(2000):
    optimizer.minimize(compute_loss,var_list=[a,b,c])
    if(i%100==99):
        print(i,'a:',a.numpy(),' b:',b.numpy(),' c:',c.numpy(),' loss:',compute_loss().numpy())
line_x = np.arange(min(x),max(x),0.01)
line_y = a*line_x*line_x+b*line_x+c

plt.plot(line_x,line_y,'-r')
plt.plot(x,y,'bo')
plt.show()


#NNregressiong
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='tanh', input_shape=[1,]),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07),loss='MSE')
model.summary()

model.fit(x,y,epochs=30)
predict = model.predict(x)

x_line = np.arange(min(x),max(x),0.01)
y_line = model.predict(x_line)
plt.plot(x_line,y_line,'r-')
plt.plot(x,y,'bo')
plt.show()