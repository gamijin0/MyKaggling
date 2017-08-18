import tensorflow as tf
from progressbar import ProgressBar
from mydataset.mydataset import MyDataSet,cleanLogdir
import numpy as np
import os

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def saveAnswer(y_):
    with open("answer.txt",mode='w') as f:
        f.write("ImageId,Label\n")
        for i,v in enumerate(y_):
            f.write("%d,%d\n" % (i+1,int(np.argmax(v))))

#用于初始化权重和偏置项
def weight_variable(shape,name=None):
  initial = tf.truncated_normal(shape, stddev=0.1) #正态分布随机
  return tf.Variable(initial,name=name)
def bias_variable(shape,name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


if (__name__ == "__main__"):
    cleanLogdir('mnist_logs')
    train_set = MyDataSet('data/train.csv')


    sess = tf.InteractiveSession()  # 创建会话

    x = tf.placeholder("float", name="Input", shape=[None, 784])  # 图片输入  420000*784
    y_ = tf.placeholder("float", name="Lable", shape=[None, 10])  # 正确的结果(用于和y比较) 420000*10

    # W = tf.Variable(tf.zeros([784, 10]), name="Weights")  # W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
    # b = tf.Variable(tf.zeros([10]), name="Bias")  # bias




    w1  =weight_variable([784,100],name="W1")
    b1 = bias_variable([100],name="B1")

    w2 = weight_variable([100,30],name="W2")
    b2 = bias_variable([30],"B2")

    w3 = weight_variable([30, 10],"W3")
    b3  =weight_variable([10],"B3")

    a1 = tf.nn.sigmoid(tf.matmul(x,w1)+b1,name="A1")

    a2 = tf.nn.sigmoid(tf.matmul(a1, w2) + b2, name="A2")

    y = tf.nn.softmax(tf.matmul(a2,w3)+b3,name="Predict")

    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_, name="Cost"))

    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    tf.summary.scalar("Cost", cross_entropy)  # 持久化一个标量Cost
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('mnist_logs', sess.graph)


    lable = tf.one_hot(train_set.data.iloc[:, 0], 10)

    sess.run(tf.global_variables_initializer())  # 初始化变量

    step_length = 200
    Train_times = 4000
    print("Step_length: %d, Train_times: %d" % (step_length, Train_times))

    with ProgressBar() as bar:
        print("Brgin fast training:")
        for i in bar(range(Train_times)):
            # 随机抓取训练数据中的100个批处理数据点，然后用这些数据点作为参数替换之前的占位符(x,y_)来运行train_step
            # 而非每次都将所有数据作为趋近目标
            batch = train_set.getNextBatch(step_length)
            this_y = sess.run(tf.one_hot(batch.iloc[:, 0], 10))
            _, summary = sess.run([train_step, merged], feed_dict={y_: this_y, x: batch.iloc[:, 1:]})
            summary_writer.add_summary(summary, i)


    print("\nBigin predicting:")
    testset = MyDataSet("data/test.csv")
    predict = sess.run(y,feed_dict={x:testset.data.iloc[:,:]})
    saveAnswer(predict)
    print("Finished.")