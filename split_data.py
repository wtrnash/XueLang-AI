import os
import shutil
import random


# 将源路径中的图片按类别分割成目标路径下的训练集和测试集
def split_data(source_path, target_path, classes):
    """
    :param source_path:  待分割的图片集的所在路径
    :param target_path:  分割完后存放训练集、测试集的所在路径
    :param classes:      所有图片类别
    """
    train_target_path = target_path + "\\训练集"
    # 目标训练集路径不存在则创建该路径
    if not os.path.exists(train_target_path):
        os.mkdir(train_target_path)
    train_target_path = train_target_path + "\\"
    test_target_path = target_path + "\\测试集"
    # 目标测试集路径不存在则创建该路径
    if not os.path.exists(test_target_path):
        os.mkdir(test_target_path)
    test_target_path = test_target_path + "\\"

    # 在训练集、测试集目录下创建对应类别文件夹
    for name in classes:
        train_class_path = train_target_path + name
        test_class_path = test_target_path + name
        if not os.path.exists(train_class_path):
            os.mkdir(train_class_path)
        if not os.path.exists(test_class_path):
            os.mkdir(test_class_path)

    source_path = source_path + "\\"
    for name in classes:
        print(name)
        class_path = source_path + name
        if not os.path.exists(class_path):
            return
        for image_name in os.listdir(class_path):
            if image_name.endswith('jpg'):
                image_path = class_path + "\\" + image_name
                # 随机产生1~5的整数，因为要按9 ： 1的比率随机分出训练集和测试集
                # 所以生成1~5的随机数，如果是1，则分到测试集，否则分到训练集
                random_number = random.randint(1, 10)
                if random_number == 1:
                    shutil.copyfile(image_path,  test_target_path + name + "\\" + image_name)
                else:
                    shutil.copyfile(image_path, train_target_path + name + "\\" + image_name)
    return


cloth_classes = ['正常','扎洞', '毛斑', '擦洞', '毛洞', '织稀', '吊经', '缺经', '跳花', '油污渍', '其他']  # 要分类图像类别
split_data("./data/瑕疵数据集", "./data/分割", cloth_classes)
