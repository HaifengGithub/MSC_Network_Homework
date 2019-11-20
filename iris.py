import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split


df = pd.read_csv('iris.csv')
species_list = ['setosa','versicolor','virginica']
x = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].values
x_data = x/x.max()
y_data_label = np.array([species_list.index(index_name) for index_name in df['Species'].values])
y_data = np.array([[int(k) for k in np.arange(3) == i] for i in y_data_label])

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

def add_layer(input,input_size,out_size,activate_function=None):
    Weights = tf.Variable(tf.truncated_normal(shape=[input_size,out_size]))
    bias = tf.Variable(tf.zeros(shape=[out_size])+0.1)
    output = tf.matmul(input, Weights)+bias
    if activate_function==None:
        return output
    else:
        return activate_function(output)

xs = tf.placeholder(shape=[None,4],dtype=tf.float32)
ys = tf.placeholder(shape=[None,3],dtype=tf.float32)
l1 = add_layer(xs,4,30,tf.nn.relu)
l2 = add_layer(l1,30,10,tf.nn.relu)
l3 = add_layer(l2,10,3)
prediction = tf.nn.softmax(l3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

init = tf.global_variables_initializer()

loss_recorder = []
iterations = []
train_index = list(range(len(y_train)))
test_index = list(range(len(y_test)))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(optimizer,feed_dict={xs: X_train,ys:y_train})
        if i % 1 == 0:
            myloss = sess.run(cross_entropy,feed_dict={xs:X_train,ys:y_train})
            loss_recorder.append(myloss)
            iterations.append(i)
            print(myloss)
	
    plt.plot(iterations, loss_recorder)
    plt.title('loss curve')
    plt.show()