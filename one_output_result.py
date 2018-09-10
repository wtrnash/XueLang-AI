import tensorflow as tf
import os
import pandas as pd
import numpy as np
import resnet_v1
import tensorflow.contrib.slim as slim
np.set_printoptions(suppress=True)
defect_classes = ['norm', 'defect_1', 'defect_2', 'defect_3', 'defect_4', 'defect_5', 'defect_6',
                  'defect_7', 'defect_8', 'defect_9', 'defect_10']


def read_and_decode(filename, size):
    # 根据文件名生成一个队列
    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'file_name': tf.FixedLenFeature([], tf.string),
                                               'image_byte': tf.FixedLenFeature([], tf.string)
                                           })
        image = tf.decode_raw(features['image_byte'], tf.uint8)
        image = tf.reshape(image, [size, size, 3])
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)  # 将图像标准化，有利于加速训练
        file_name = features['file_name']

    return image, file_name


# 递归计算图片总数
def count_image_number(path):
    count = 0
    for index, file_name in enumerate(os.listdir(path)):
        file_path = path + file_name
        # 如果是文件夹则递归调用本函数
        if os.path.isdir(file_path):
            count = count + count_image_number(file_path + '\\')
        else:
            count = count + 1
    return count


# 科学计数法转为浮点型
def as_num(num):
    float_num = '{:.5f}'.format(num)
    if float(float_num) > float(0.99999):
        float_num = float(0.99999)
    if float(float_num) < float(0.00001):
        float_num = float(0.00001)
    return float(float_num)


path = './data/xuelang_round2_test_b_201808031/'
# 计算图片数量
count = count_image_number(path)
batch_size = 20

# 声明占位符
x = tf.placeholder(tf.float32, [None, 400, 400, 3], name="x")

# 获得网络结果
# logits = model.model(x, is_training=False, dropout_pro=0.5, num=len(defect_classes), weight_decay=0.0)
net, _ = resnet_v1.resnet_v1_50(x, is_training=False)
net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
logits = slim.fully_connected(net, num_outputs=len(defect_classes),
                              activation_fn=None, scope='predict')
# logits, _ = inception_v3.inception_v3(x, len(defect_classes), is_training=False)

logits = tf.nn.softmax(logits)
logits = tf.clip_by_value(logits, 1e-10, 1)

# 初始化回话
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './checkpoint/resnet50_0.558/resnet_3000.ckpt')
tf_path = "./classification_tf/400classification.TFRecord"

image, file_name = read_and_decode(tf_path, 400)

image_batch, file_name_batch = tf.train.batch(
    [image, file_name], batch_size=batch_size, capacity=3000, allow_smaller_final_batch=True)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

defect_list = []
i = 0
while i + batch_size <= count:
    i = i + batch_size

    image_value, file_name_value = sess.run([image_batch, file_name_batch])
    value = sess.run(logits, feed_dict={x: image_value})  # 传入网络得到结果
    for j in range(batch_size):
        for index, _ in enumerate(defect_classes):
            defect_list.append([file_name_value[j].decode() + "|" + defect_classes[index], as_num(value[j][index])])


# 处理最后一批数据
image_value, file_name_value = sess.run([image_batch, file_name_batch])
value = sess.run(logits, feed_dict={x: image_value})  # 传入网络得到结果

for j in range(count - i):
    for index, _ in enumerate(defect_classes):
        defect_list.append([file_name_value[j].decode() + "|" + defect_classes[index], as_num(value[j][index])])

coord.request_stop()
coord.join(threads)

        
result_df = pd.DataFrame(defect_list)  # 转为DataFrame
result_df = result_df.drop_duplicates()  # 去重
result_df.rename(columns={0: 'filename|defect', 1: 'probability'}, inplace=True)
result_df.to_csv("resnet152_submit.csv", encoding='utf-8', index=None)
