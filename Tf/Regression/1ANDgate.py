import tensorflow as tf
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x) )


x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[0],[0],[1]])

w = tf.random.normal([2],0,1)
b = tf.random.normal([1],0,1)

print(x.shape,w)
print(y.shape,b)
print(x[0],x[0][0])
b_x= 1 #base bias
#1node inputs
for i in range(3000):
    error_sum = 0
    for j in range(4):
        output=sigmoid(np.sum(x[j]*w)+b_x*b)
        error=y[j][0] - output
        error_sum += error
        w = w + x[j]*0.1*error
        b = b + b_x*0.1*error
    if i%200 == 199:
        print(i,error_sum)
print(w)
print(b)

#test!
for i in range(4):
    print("X :",x[i]," Y :",y[i]," Output :", sigmoid(np.sum(x[i]*w)+b))