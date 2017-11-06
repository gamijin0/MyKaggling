# data analysis and wrangling
import pandas as pd
import numpy as np
from progressbar import ProgressBar
from mydataset.mydataset import MyDataSet, cleanLogdir

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# machine learning
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =======================变量配置=====================================

USE_SAVE=True

save_file_path = "../save/model.ckpt"
save_dir_path = "../save"
log_path = "../log"
train_file_path = "../data/train.csv"
test_file_path = "../data/test.csv"

# 每层节点数
hidden_nodes_1 = 57  # num of features
hidden_nodes_2 = 100

hidden_nodes_3 = 100
output_nodes = 1

learning_rate = 0.0001

beta = 0.01

step_length = 10000
Train_times = 10000


# =======================自定义函数====================================


def normalize(series):
    mean = series.mean()
    stdev= series.std()
    return (series - mean)/stdev


# 用于初始化权重和偏置项
def weight_variable(shape, name=None):
    initial = tf.random_normal(shape, stddev=1)  # 正态分布随机
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.truncated_normal(shape=shape) #截断正态随机
    return tf.Variable(initial, name=name)


# 判断预测的精准程度
def accuracy(predictions, labels):
    predictions = predictions > 0.5
    return (np.sum(predictions == labels) / predictions.shape[0])


# 从一个batch中提取lable
def getLable(batch):
    return batch.iloc[:, 1].values.reshape(batch.shape[0], 1)


def getFeatures(batch):
    return batch.iloc[:, 2:].values

def process_data(dataFrame):
    # col_to_drop = dataFrame.columns[dataFrame.columns.str.startswith('ps_calc_')]
    # dataFrame = dataFrame.drop(col_to_drop,axis=1)

    dataFrame.replace(-1,np.nan)

    to_be_normalized = dataFrame.iloc[:,2:]
    dataFrame.loc[:,2:] = (to_be_normalized-to_be_normalized.mean())/(to_be_normalized.max()-to_be_normalized.min())

    cat_features = [a for a in dataFrame.columns if a.endswith('cat')]
    for column in cat_features:
        temp = pd.get_dummies(pd.Series(dataFrame[column]))
        train = pd.concat([dataFrame, temp], axis=1)
        train = dataFrame.drop([column], axis=1)
    return dataFrame


# =======================读取数据====================================
cleanLogdir(log_path)

train_set = MyDataSet(train_file_path, header=0)
train_set.setData(process_data(train_set.data))
challange = MyDataSet(test_file_path, header=0)
challange.setData(process_data(challange.data))

ids = challange.data.iloc[:, 0].values
final = pd.DataFrame({
    'id': ids,
    'target': np.zeros(ids.shape[0])
})

# =======================TensorFlow==================================



sess = tf.Session(config=tf.ConfigProto(device_count={'gpu': 0}))

# 占位符,运行时由实际数据替代
x = tf.placeholder(dtype="float64", shape=[None, hidden_nodes_1], name="Input")
y_ = tf.placeholder(dtype="float64", shape=[None, output_nodes], name="Lable")

# 定义每层的weight 与 bias
# weight使用truncated_normal(截断正态随机)初始化,bias填充0
weight1 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], dtype=tf.float64), name="W1")
biases1 = tf.Variable(tf.zeros([hidden_nodes_2], dtype=tf.float64), name="B1")
m1 = tf.matmul(x, weight1) + biases1  # 数据入口x
res1 = tf.nn.relu(m1)
res1 = tf.nn.dropout(res1,0.1)

weight2 = tf.Variable(tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], dtype=tf.float64), name="W2")
biases2 = tf.Variable(tf.zeros([hidden_nodes_3], dtype=tf.float64), name="B2")
m2 = tf.matmul(res1, weight2) + biases2
res2 = tf.nn.relu(m2)
res2 = tf.nn.dropout(res2,0.01)

weight3 = tf.Variable(tf.truncated_normal([hidden_nodes_3, output_nodes], dtype=tf.float64), name="W3")
biases3 = tf.Variable(tf.zeros([output_nodes], dtype=tf.float64), name="B3")
res3 = tf.matmul(res2, weight3) + biases3

# regularizers = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2) + tf.nn.l2_loss(weight3)  #用于避免过拟合
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=res3, labels=y_, name="Cost"))  # 结果出口y
# loss = tf.reduce_mean(loss + beta * regularizers)

y = tf.nn.sigmoid(res3)

# 优化目标
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.02).minimize(loss)

tf.summary.scalar("Cost", loss)  # 持久化一个标量Cost
merged = tf.summary.merge_all()  # 用于可视化
summary_writer = tf.summary.FileWriter(log_path, sess.graph)

saver = tf.train.Saver()  # 用于保存模型
# ======保存与读取模型=========

if (os.path.exists(save_dir_path) and USE_SAVE==True):
    # 若有存档则读取存档
    saver.restore(sess, save_file_path)
    print("Restored")
else:
    sess.run(tf.global_variables_initializer())  # 初始化变量
    print('Initialized')
    if(not os.path.exists(save_dir_path)):
        os.mkdir(save_dir_path)

print("Step_length: %d, Train_times: %d" % (step_length, Train_times))

with ProgressBar() as bar:  # 进度条
    print("Begin  training:")
    for i in bar(range(Train_times)):
        # 随机抓取训练数据中的step_length个批处理数据点，然后用这些数据点作为参数替换之前的占位符(x,y_)来运行train_step
        # 而非每次都将所有数据作为趋近目标
        batch = train_set.getNextBatch(step_length)
        _, summary,cost = sess.run([optimizer, merged,loss], feed_dict={x: getFeatures(batch), y_: getLable(batch)})
        summary_writer.add_summary(summary, i)
        if (i % (Train_times / 100) == 0):
            saver.save(sess, save_file_path)  # 每达到进度的1%就保存一次
            print(" [cost]:",cost)
            #     test_batch = train_set.getNextBatch(step_length)
            #
            #     predict = sess.run(y,feed_dict={x:getFeatures(test_batch)})
            #     print('Training accuracy: %f' % accuracy(predict, getLable(test_batch)))
            #
        if(i%(Train_times/2)==0):
            # 保存结果
            final_prediction = sess.run(y, feed_dict={x: challange.data.iloc[:, 1:].values})
            final["target"] = final_prediction
            final.to_csv("prediction.csv", index=False, float_format='%.5f')
            print("Make prefiction...")
