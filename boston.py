import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split

Boston = datasets.load_boston()
x = Boston.data
y= Boston.target[:,np.newaxis]
x_data = x/x.max()
y_data = y/y.max()

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

def add_layer(input,input_size,out_size,activate_function=None):
    Weights = tf.Variable(tf.truncated_normal(shape=[input_size,out_size]))
    bias = tf.Variable(tf.zeros(shape=[out_size])+0.1)
    output = tf.matmul(input, Weights)+bias
    if activate_function==None:
        return output
    else:
        return activate_function(output)

xs = tf.placeholder(shape=[None,13],dtype=tf.float32)
ys = tf.placeholder(shape=[None,1],dtype=tf.float32)
l1 = add_layer(xs,13,30,tf.nn.relu)
l2 = add_layer(l1,30,10,tf.nn.relu)
predict = add_layer(l2,10,1)

loss = tf.reduce_mean((ys-predict)**2)

optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()

loss_recorder = []
iterations = []
train_index = list(range(len(y_train)))
test_index = list(range(len(y_test)))


with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(optimizer,feed_dict={xs:X_train,ys:y_train})
        if i % 1 == 0:
            myloss = sess.run(loss,feed_dict={xs:X_test,ys:y_test})
            loss_recorder.append(myloss)
            iterations.append(i)
            print(myloss)
    y_train_predict = sess.run(predict, feed_dict={xs: X_train})
    y_test_predict = sess.run(predict, feed_dict={xs: X_test})
    plt.plot(iterations, loss_recorder)
    plt.show()

    plt.scatter(train_index, y_train, color='red')

    plt.scatter(train_index, y_train_predict, color='blue')
    plt.title("training set")
    plt.show()

    plt.scatter(test_index, y_test, color='red')


    plt.scatter(test_index, y_test_predict, color='blue')
    plt.title("validation set")
    plt.show()