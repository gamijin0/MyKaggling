import tensorflow as tf
from progressbar import ProgressBar
bar = ProgressBar()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession() #创建会话

x = tf.placeholder("float", shape=[None, 784])  #图片输入
W = tf.Variable(tf.zeros([784,10]))# W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
b = tf.Variable(tf.zeros([10])) #bias
y = tf.nn.softmax(tf.matmul(x,W) + b)  #模型的具体描述 : y为预测结果



y_ = tf.placeholder("float", shape=[None, 10])  #正确的结果(用于和y比较)
#成本函数:交叉熵
#tf.reduce_sum 计算张量的所有元素的总和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#指定训练模式:梯度下降
#指定成本函数为:cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


sess.run(tf.global_variables_initializer()) #初始化变量
for i in bar(range(1000)):
    #随机抓取训练数据中的100个批处理数据点，然后用这些数据点作为参数替换之前的占位符(x,y_)来运行train_step
    #而非每次都将所有数据作为趋近目标
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

