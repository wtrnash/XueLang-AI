import tensorflow as tf
import os
from PIL import Image


def image_to_tf(source_path, target_path, classes, name):
    with tf.python_io.TFRecordWriter(target_path + "\\" + name) as writer:  # 用来生成训练集TFRecord格式文件
        for index, name in enumerate(classes):
            class_path = source_path + "\\" + name + "\\"  # 训练集中每个类的地址
            if not os.path.exists(class_path):
                return
            for image_name in os.listdir(class_path):
                if image_name.endswith('jpg'):
                    image_path = class_path + image_name  # 每个图片的地址
                    image = Image.open(image_path)  # 打开图片
                    image = image.resize((400, 400), Image.ANTIALIAS)  # 将图片大小统一设为 600 * 600
                    image_byte = image.tobytes()  # 图片转换为二进制格式
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'image_byte': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte]))
                    }))  # 通过Example将图像和标签封装
                    writer.write(example.SerializeToString())   # 序列化为字符串

    return


cloth_classes = ['正常', '扎洞', '毛斑', '擦洞', '毛洞', '织稀', '吊经', '缺经', '跳花', '油污渍', '其他']  # 要分类图像类别
image_to_tf("./data/分割/训练集", "./data/转换", cloth_classes, "400train.TFRecord")
image_to_tf("./data/分割/测试集", "./data/转换", cloth_classes, "400test.TFRecord")