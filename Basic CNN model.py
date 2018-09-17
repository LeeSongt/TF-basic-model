
# coding: utf-8

# In[1]:


import sys

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# In[2]:


data_dir = "./data"
mnist = input_data.read_data_sets(data_dir, one_hot=True)


# In[3]:


CONV_1_SIZE = 3
CONV_1_DEEP = 32
INPUT_CHANNELS = 1

CONV_2_SIZE = 3
CONV_2_DEEP = 64

BATCH_SIZE = 100

LEARNING_RATE_INIT = 1e-3


# In[4]:


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


# In[5]:


with tf.variable_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])


# In[6]:


with tf.variable_scope("conv1"):
    initial_value = tf.truncated_normal([CONV_1_SIZE, CONV_1_SIZE, INPUT_CHANNELS, CONV_1_DEEP], stddev=0.1)
    conv_1_w = tf.Variable(initial_value=initial_value, collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    conv_1_b = tf.Variable(initial_value=tf.constant(0.1, shape=[CONV_1_DEEP]))
    conv_1_l = tf.nn.conv2d(x_image, conv_1_w, strides=[1,1,1,1], padding='SAME') + conv_1_b
    conv_1_h = tf.nn.relu(conv_1_l)
    print(conv_1_l.shape)


# In[7]:


with tf.variable_scope("pool1"):
    pool_1_h = tf.nn.max_pool(conv_1_h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print(pool_1_h.shape)


# In[8]:



with tf.variable_scope('conv2'):
    conv_2_w = tf.Variable(tf.truncated_normal([CONV_2_SIZE,CONV_2_SIZE,CONV_1_DEEP,CONV_2_DEEP], stddev=0.1),
                           collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    conv_2_b = tf.Variable(tf.constant(0.1, shape=[CONV_2_DEEP]))
    conv_2_l = tf.nn.conv2d(pool_1_h, conv_2_w, strides=[1,1,1,1], padding='SAME') + conv_2_b
    conv_2_h = tf.nn.relu(conv_2_l)
    print(conv_2_h.shape)


# In[9]:


with tf.name_scope('pool2'):
    pool_2_h = tf.nn.max_pool(conv_2_h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print(pool_2_h.shape)


# In[10]:


with tf.name_scope('fc1'):
    #
    fc_1_w = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
    fc_1_b = tf.Variable(tf.constant(0.1, shape=[1024]))
    #全连接层的输入为向量,而池化层2的输出为7x7x64的矩阵,所以这里要将矩阵转化成一个向量
    pool_2_h_flat = tf.reshape(pool_2_h, [-1,7*7*64])
    fc_1_h = tf.nn.relu(tf.matmul(pool_2_h_flat, fc_1_w) + fc_1_b)
    print(fc_1_h.shape)


# In[11]:


with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    fc_1_h_drop = tf.nn.dropout(fc_1_h, keep_prob)


# In[12]:


with tf.name_scope('fc2'):
    fc_2_w = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1), collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])
    fc_2_b = tf.Variable(tf.constant(0.1, shape=[10]))
    y = tf.matmul(fc_1_h_drop, fc_2_w) + fc_2_b


# In[13]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# In[14]:


l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection('WEIGHTS')])


# In[15]:


total_loss = cross_entropy + 7e-5*l2_loss


# In[16]:


train_step = tf.train.AdamOptimizer(LEARNING_RATE_INIT).minimize(total_loss)


# In[17]:


sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)
 
#Train
for step in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _, loss, l2_loss_value, total_loss_value = sess.run(
        [train_step, cross_entropy, l2_loss, total_loss],
        feed_dict={x: batch_xs, y_:batch_ys, keep_prob:0.5})
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #
    if (step+1)%200 == 0:
        #每隔200步评估一下训练集和测试集
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
        test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
        print("step:%d, loss:%f, train_acc:%f, test_acc:%f" % (step, total_loss_value, train_accuracy, test_accuracy))

