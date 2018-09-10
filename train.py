import tensorflow as tf
import resnet_v1
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
    train_image, train_label = read_and_decode("./data/转换/400train.TFRecord")

    train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=20,
                                                                  capacity=4000, min_after_dequeue=2000,
                                                                  allow_smaller_final_batch=True)

    x = tf.placeholder(tf.float32, [None, 400, 400, 3], name="x")
    y_ = tf.placeholder(tf.int64, [None], name="y_")
    tf.summary.image("input_image", x, 10)

    net, _ = resnet_v1.resnet_v1_152(x)
    net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
    logits = slim.fully_connected(net, num_outputs=len(classes),
                                  activation_fn=None, scope='predict')

    # 计算交叉熵及其平均值
    with tf.name_scope('training'):
        labels = tf.one_hot(y_, len(classes))
        logits = tf.nn.softmax(logits)
        logits = tf.clip_by_value(logits, 1e-10, 1.0)
        loss = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(logits)))
        # 优化损失函数
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(y_, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 初始化回话并开始训练过程。
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train_model_demo', sess.graph)
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, './checkpoint/resnet.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 循环的训练神经网络

        for i in range(15000):
            img, lbl = sess.run([train_image_batch, train_label_batch])
            sess.run(train_step, feed_dict={x: img, y_: lbl})
            if i % 10 == 0:
                summary, l, acc = sess.run([merged, loss, accuracy], feed_dict={x: img, y_: lbl})
                print("After %d training step(s), loss is %g ,accuracy is %g" % (i, l, acc))
                train_writer.add_summary(summary, i)
            if i % 3000 == 0 and i != 0 and i <= 15000:
                saver.save(sess, './checkpoint/resnet_' + str(i) + '.ckpt')

        saver.save(sess, './checkpoint/resnet.ckpt')

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
