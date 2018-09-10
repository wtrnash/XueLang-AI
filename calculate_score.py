import resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim

classes = ['正常', '扎洞', '毛斑', '擦洞', '毛洞', '织稀', '吊经', '缺经', '跳花', '油污渍', '其他']  # 要分类图像类别


def read_and_decode(filename):
    # 根据文件名生成一个队列
    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_byte': tf.FixedLenFeature([], tf.string),
                                           })
        image = tf.decode_raw(features['image_byte'], tf.uint8)
        image = tf.reshape(image, [400, 400, 3])
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)  # 将图像标准化，有利于加速训练
        label = tf.cast(features['label'], tf.int32)

    return image, label


def main(_):
    test_image, test_label = read_and_decode("./data/转换/400test.TFRecord")

    test_img_batch, test_lbl_batch = tf.train.batch([test_image, test_label], batch_size=1, capacity=2000)

    x = tf.placeholder(tf.float32, [None, 400, 400, 3], name="x")
    y_ = tf.placeholder(tf.int64, [None], name="y_")
    tf.summary.image("input_image", x, 10)

    net, _ = resnet_v1.resnet_v1_152(x, is_training=False)
    net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
    logits = slim.fully_connected(net, num_outputs=len(classes),
                                  activation_fn=None, scope='predict')

    # 计算交叉熵及其平均值
    with tf.name_scope('training'):
        labels = tf.one_hot(y_, len(classes))
        logits = tf.nn.softmax(logits)
        logits = tf.clip_by_value(logits, 1e-10, 1.0)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(logits)))
        # 损失函数的计算
        loss = cross_entropy
        # 优化损失函数
        tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(y_, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 初始化回话并开始训练过程。
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './checkpoint/resnet_15000.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        map_label_list = []
        map_prediction_list = []

        auc_label_list = []
        auc_prediction_list = []
        # 342是样本数
        for i in range(342):
            test_img, test_lbl = sess.run([test_img_batch, test_lbl_batch])
            temp, prediction = sess.run([accuracy, logits], feed_dict={x: test_img, y_: test_lbl})
            map_label_list.append(test_lbl[0])
            map_prediction_list.append(list(prediction[0]))

            if test_lbl[0] == 0:
                auc_label_list.append(True)
            else:
                auc_label_list.append(False)

            auc_prediction_list.append(prediction[0][0])

        prediction_tensor = tf.convert_to_tensor(auc_prediction_list)
        label_tensor = tf.convert_to_tensor(auc_label_list)
        map_prediction_tensor = tf.convert_to_tensor(map_prediction_list, dtype=tf.float32)
        map_label_tensor = tf.convert_to_tensor(map_label_list, dtype=tf.int64)
        auc_value, auc_op = tf.metrics.auc(label_tensor, prediction_tensor, num_thresholds=400)
        map_value, map_op = tf.metrics.average_precision_at_k(map_label_tensor, map_prediction_tensor, 1)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run([auc_op, map_op])
        auc, mAP = sess.run([auc_value, map_value])

        print("AUC:" + str(auc))
        print("mAP:" + str(mAP))
        score = 0.7 * auc + 0.3 * mAP
        print("score:" + str(score))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
