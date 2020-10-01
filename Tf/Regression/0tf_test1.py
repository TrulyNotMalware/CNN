import tensorflow as tf
import numpy as np

@tf.function
def add_1_or_10(x,b):
    if b:
        x += 1.
    else:
        x+= 10.
    return x

@tf.function
def gradient_decent(x,y):
    w = tf.random.normal([1],0,1)
    print(w)
    for i in range(1000):
        output = tf.sigmoid(x*w)
        error = output - y
        w=w+x*0.1*error
        if(i%100 == 99):
            print(i,error,output)
    return output
result = add_1_or_10(tf.constant(1.), True).numpy()
print(result)
print(tf.__version__)

test1 = tf.random.normal([4],0,1)
print(test1)

x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[0],[0],[1]])

