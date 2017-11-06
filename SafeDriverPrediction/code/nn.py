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
prediction_file_path = "prediction.csv"

# 每层节点数
feature_num = 57  # num of features,will be changed
hidden_nodes_1 = 50

hidden_nodes_2 = 50
output_nodes = 1

MAX_ONE_HOT_SIZE = 20

learning_rate = 0.001

beta = 0.01

step_length = 6000
Train_times = 1000


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


def process_data(train_df,test_df):
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    train_df.fillna(-1) #用-1填充空缺值
    test_df.fillna(-1)
    data_col  = [c for c in train_df.columns if c not in ["id", "target"]] #找出所有的数据列的列名
    train_df['negative_one_vals'] = np.sum((train_df[data_col] == -1).values, axis=1) #将-1出现的次数也作为一个feature
    test_df['negative_one_vals'] = np.sum((test_df[data_col] == -1).values, axis=1)
    for col in data_col:
        #检查每一列,如果此列的值是由少量的离散值构成,则对此列进行one_hot编码(映射为新的feature)
        #故需要同时对test数据集进行one_hot
        unique_value = train_df[col].unique()
        if(len(unique_value)>2 and len(unique_value)<MAX_ONE_HOT_SIZE):
            for val in unique_value:
                train_df[col + '_' + str(val)] = np.int8(train_df[col].values == val)
                test_df[col+'_'+str(val)] = np.int8(test_df[col].values==val)
    return train_df,test_df

def process_data_1(df):

    df = pd.DataFrame(df)
    d_median = df.median(axis=0)
    d_mean = df.mean(axis=0)
    df.fillna(-1)
    one_hot = {c: list(df[c].unique()) for c in df.columns if c not in ['id', 'target']}
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:  # standard arithmetic
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            # df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            # df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            # df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            # df[c+str('_exp')] = np.exp(df[c].values) - 1
    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)


    to_be_normalized = df.iloc[:, 2:]
    df.loc[:, 2:] = (to_be_normalized - to_be_normalized.mean()) / (to_be_normalized.max() - to_be_normalized.min())

    return df


def predict(test_df,sess,filename):
    res = pd.DataFrame({'id':[],'target':[]})
    PREDICT_BATCH_SIZE = 50000
    test_df = pd.DataFrame(test_df)
    for i in range(0,int(test_df.values.shape[0]/PREDICT_BATCH_SIZE)+1):
        batch = test_df.iloc[i*PREDICT_BATCH_SIZE:(i+1)*PREDICT_BATCH_SIZE,1:]
        temp_prediction = pd.DataFrame({
            'id':batch.iloc[:,0].values
        })
        temp_prediction['target'] = sess.run(y, feed_dict={x: batch.values})
        res  = pd.concat([res,temp_prediction])

    res.to_csv(filename, index=False, float_format='%.5f')
    print("Make prefiction...")

# =======================读取数据====================================
cleanLogdir(log_path)

train_set = MyDataSet(train_file_path, header=0)
challange_set = MyDataSet(test_file_path, header=0)

tr_df,ch_df = process_data(train_set.data, challange_set.data)

train_set.setData(tr_df)
challange_set.setData(ch_df)

feature_num =  train_set.data.values.shape[1]-2
print("Feature num:",feature_num)


#
# ids = challange_set.data.iloc[:, 0].values
# final = pd.DataFrame({
#     'id': ids,
#     'target': np.zeros(ids.shape[0])
# })

# =======================TensorFlow==================================



sess = tf.Session(config=tf.ConfigProto(device_count={'gpu': 0}))

# 占位符,运行时由实际数据替代
x = tf.placeholder(dtype="float64", shape=[None, feature_num], name="Input")
y_ = tf.placeholder(dtype="float64", shape=[None, output_nodes], name="Lable")

# 定义每层的weight 与 bias
# weight使用truncated_normal(截断正态随机)初始化,bias填充0
weight1 = tf.Variable(tf.truncated_normal([feature_num, hidden_nodes_1], dtype=tf.float64), name="W1")
biases1 = tf.Variable(tf.zeros([hidden_nodes_1], dtype=tf.float64), name="B1")
m1 = tf.matmul(x, weight1) + biases1  # 数据入口x
res1 = tf.nn.relu(m1)
res1 = tf.nn.dropout(res1,0.1)

weight2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], dtype=tf.float64), name="W2")
biases2 = tf.Variable(tf.zeros([hidden_nodes_2], dtype=tf.float64), name="B2")
m2 = tf.matmul(res1, weight2) + biases2
res2 = tf.nn.relu(m2)
res2 = tf.nn.dropout(res2,0.01)

weight3 = tf.Variable(tf.truncated_normal([hidden_nodes_2, output_nodes], dtype=tf.float64), name="W3")
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
        if(i%(Train_times/2)==0 and i>0):
            # 保存结果
            predict(challange_set.data,sess,prediction_file_path)