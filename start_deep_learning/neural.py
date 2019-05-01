import numpy as np
import matplotlib.pylab as plt
import sys, os


def step_function(x):
    return np.array( x > 0, dtype = np.int)

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)  # overflow 방지
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def relu(x):
    return np.maximum( 0, x )


## test code ##

#a = np.array([0.3, 2.9, 4.0])
#y = softmax(a)
#print(y)
#print(np.sum(y))
#print( step_function(a))

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x,y)
plt.show()



