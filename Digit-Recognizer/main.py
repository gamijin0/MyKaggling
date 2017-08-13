import tensorflow as tf
from progressbar import ProgressBar
from mydataset.mydataset import MyDataSet
import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def saveAnswer(y_):
    with open("answer.txt",mode='w') as f:
        f.write("ImageId,Label\n")
        for i,v in enumerate(y_):
            f.write("%d,%d\n" % (i+1,np.argmax(v)))


if (__name__ == "__main__"):

    train_set = MyDataSet('data/train.csv')

    sess = tf.InteractiveSession()  # 创建会话

    x = tf.placeholder("float", name="Input", shape=[None, 784])  # 图片输入  420000*784
    y_ = tf.placeholder("float", name="Lable", shape=[None, 10])  # 正确的结果(用于和y比较) 420000*10

    W = tf.Variable(tf.zeros([784, 10]), name="Weights")  # W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
    b = tf.Variable(tf.zeros([10]), name="Bias")  # bias

    y = tf.nn.softmax(tf.matmul(x, W) + b, name="Predict")  # 模型的具体描述 : y为预测结果

    with tf.name_scope('Cost'):
        # 成本函数:交叉熵
        # tf.reduce_sum 计算张量的所有元素的总和
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_,name="Cost"))
        # cross_entropy = -tf.reduce_sum(y_*tf.log(y),name="Cost")
        # cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

    with tf.name_scope('Train'):
        # 指定训练模式:梯度下降
        # 指定成本函数为:cross_entropy
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # tf.summary.scalar("Accuracy", accuracy)  # 持久化一个标量Cost

    tf.summary.scalar("Cost", cross_entropy)  # 持久化一个标量Cost

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('mnist_logs', sess.graph)

    lable = tf.one_hot(train_set.data.iloc[:, 0], 10)

    sess.run(tf.global_variables_initializer())  # 初始化变量

    step_length = 500
    Train_times = 2000
    print("Step_length: %d, Train_times: %d" % (step_length, Train_times))



    with ProgressBar() as bar:
        print("Brgin training:")
        for i in bar(range(Train_times)):
            # 随机抓取训练数据中的100个批处理数据点，然后用这些数据点作为参数替换之前的占位符(x,y_)来运行train_step
            # 而非每次都将所有数据作为趋近目标
            # batch = tf.slice(train_set,begin=[i*step_length,i*step_length],size=[step_length,step_length])
            batch = train_set.getNextBatch(step_length)
            this_y = sess.run(tf.one_hot(batch.iloc[:, 0],10))
            _, summary = sess.run([train_step, merged], feed_dict={y_:this_y , x: batch.iloc[:, 1:]})
            # summary_writer.add_summary(summary, i)
            # if (i % 100 == 0):
            #
            #     print("\nAccuracy:", sess.run(accuracy, feed_dict={y_:sess.run(lable) , x: train_set.data.iloc[:, 1:]}))
            #     # print("Cost:", sess.run(cross_entropy, feed_dict={x: train_set[0], y_: train_set[1]}))
                # print(accuracy.eval(feed_dict={x:train_set[0], y_: train_set[1]}))


    testset = MyDataSet("data/test.csv")
    predict = sess.run(y,feed_dict={x:testset.data.iloc[:,:]})
    saveAnswer(predict)