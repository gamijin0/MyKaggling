# data analysis and wrangling
import pandas as pd
import numpy as np
from progressbar import ProgressBar

# machine learning
import tensorflow as tf
from sklearn.model_selection import train_test_split


#=======================自定义函数====================================

# 将性别的string映射为0和1
def parseSex(df):
    df.loc[df['Sex'] == 'male', 'Sex'] = 1
    df.loc[df['Sex'] == 'female', 'Sex'] = 0
    return df


# 新增一个featur为FamilySize
def addFamilySize(df):
    df['FamilySize'] = 0
    df['FamilySize'] = df['Parch'] + df['SibSp']
    return df


# 归一化,有利于提高精度和加速梯度下降
def normalize(series):
    mean = series.mean()
    stdev = series.std()
    return (series - mean) / stdev


# 判断预测的精准程度
def accuracy(predictions, labels):
    predictions = predictions > 0.5
    return (np.sum(predictions == labels) / predictions.shape[0])

#预处理数据
def preprocess_data(dataFrame):
    dataFrame = parseSex(dataFrame)
    dataFrame = addFamilySize(dataFrame)

    #取平均值填充空缺值
    age_median = dataFrame["Age"].median()
    dataFrame["Age"] = dataFrame["Age"].fillna(age_median)

    return  dataFrame




#=======================读取数据====================================
train = pd.read_csv("../data/train.csv")
challange = pd.read_csv('../data/test.csv')

train = preprocess_data(train)
_,test = train_test_split(train,test_size=0.4)
challange = preprocess_data(challange)

feature_names =['Pclass', 'Sex', 'Age', 'SibSp','Parch','FamilySize']

#提取需要的数个features
train_features = np.float64(train[feature_names].values)
train_target = np.float64(train['Survived'].values)
train_target = train_target.reshape(train_target.shape[0],1)
test_features = np.float64(test[feature_names].values)
test_target = np.float64(test['Survived'].values)
test_target = test_target.reshape(test_target.shape[0],1)
challange_features  = np.float64(challange[feature_names].values)

ids = challange.index.values+892
final = pd.DataFrame({
    'PassengerId': ids,
    'Survived': np.zeros(ids.shape[0])
})


#=======================TensorFlow==================================



#每层节点数
hidden_nodes_1 = 6  #['Pclass', 'Sex', 'Age', 'SibSp','','FamilySize']
hidden_nodes_2 = 20
hidden_nodes_3 = 20
output_nodes = 1


beta =0.01

sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))



# #由于数据量较小,一次性读入所有数据
# tf_train_dataset = tf.constant(train_features)
# tf_train_lables = tf.constant(train_target)


#占位符,运行时由实际数据替代
x = tf.placeholder(dtype="float64",shape=[None,hidden_nodes_1],name="Input")
y_ = tf.placeholder(dtype="float64",shape=[None,output_nodes],name="Lable")

#定义每层的weight 与 bias
#weight使用truncated_normal(截断正态随机)初始化,bias填充0
weight1 = tf.Variable(tf.truncated_normal([hidden_nodes_1,hidden_nodes_2],dtype=tf.float64),name="W1")
biases1 = tf.Variable(tf.zeros([hidden_nodes_2],dtype=tf.float64),name="B1")
m1 = tf.matmul(x,weight1)+biases1  #数据入口x
res1 = tf.nn.relu(m1)

weight2 = tf.Variable(tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], dtype=tf.float64),name="W2")
biases2 = tf.Variable(tf.zeros([hidden_nodes_3],dtype=tf.float64),name="B2")
m2 = tf.matmul(res1,weight2)+biases2
res2 = tf.nn.relu(m2)

weight3 = tf.Variable(tf.truncated_normal([hidden_nodes_3, output_nodes], dtype=tf.float64),name="W3")
biases3 = tf.Variable(tf.zeros([output_nodes],dtype=tf.float64),name="B3")
res3 = tf.matmul(res2,weight3)+biases3

regularizers = tf.nn.l2_loss(weight1)+tf.nn.l2_loss(weight2)+tf.nn.l2_loss(weight3)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=res3,labels=y_,name="Cost")) #结果出口y
loss = tf.reduce_mean(loss+beta*regularizers)
y = tf.nn.sigmoid(res3)

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess.run(tf.global_variables_initializer())  # 初始化变量
print('Initialized')



train_steps = 100000

with ProgressBar() as bar:
    for i in bar(range(train_steps)):
        sess.run([optimizer],feed_dict={x:train_features,y_:train_target})
        if(i%1000==0):
            test_prediction =  sess.run(y,feed_dict={x:test_features})
            print('Training accuracy: %f' % accuracy(test_prediction, test_target))


final_prediction  = sess.run(y,feed_dict={x:challange_features})
final["Survived"] = np.int64(final_prediction>=0.5)

final.to_csv("titanic_predictions2.csv", index=False)

