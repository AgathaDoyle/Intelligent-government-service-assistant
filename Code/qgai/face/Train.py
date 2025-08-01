# ================================
# @File         : Train_v1.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 对输入的图片进行预处理以及模型训练（贝叶斯分类版本）
#                 主函数cv2_train(imgs_bin, user_id, min_acc=0.6)
#                 传入二进制图片数组、用户ID以及准确率阀值
#                 如果模型训练成功则返回True
# ================================

import os
import time
from multiprocessing import Pool
from os import cpu_count
import random
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Predict import single_face_image_predict
from model_loader import get_components
from utils.face_utils import (
    extract_features, normalize_features
)

components = get_components()
logger = components['logger']
paths = components['paths']
models = components['models']

knn_model_path = paths['knn_model_path']
le_path = paths['le_path']
history_features_path = paths['history_features_path']
history_labels_path = paths['history_labels_path']

# 使用模型组件
rec_model = models['rec_model']
embedder = models['embedder']
detectors = models['detectors']
classifier = models['knn_classifier']
label_encoder = models['label_encoder']


def process_single_image(face_image):
    """
    处理单张图片，用于多进程处理

    :param face_image:      人脸CV图像
    :return:                模型处理后的展平向量
    """
    try:
        features = extract_features(face_image, rec_model)  # 提取人脸特征
        features = normalize_features(features)             # 特征向量归一化

        return features

    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        return None


def get_feature_and_labels(face_images, user_id, single_pool_mode=True):
    """
    获取特征向量和对应的标签

    :param single_pool_mode:    指定是否为单线程模式，对于数量少的图片建议使用，默认开启
    :param face_images:         人脸CV图像列表
    :param user_id:             用户对应的id
    :return:                    特征数组和标签数组
    """
    logger.info(f"【提取特征向量】发现{len(face_images)}个图像文件，开始提取图像组的特征向量... ...")
    features, labels = [], []

    if single_pool_mode:
        for face_image in face_images:
            feature = process_single_image(face_image)
            if feature is not None:
                features.append(feature)
                labels.append(user_id)
    else:
        # 多线程处理图像
        num_processes = min(cpu_count(), len(face_images))
        with Pool(num_processes) as pool:
            # results = pool.map(process_single_image, images_binary)
            for feature in pool.imap(process_single_image, face_images):
                if feature is not None:
                    features.append(feature)
                    labels.append(user_id)

    return features, labels


def merge_labels(le, new_labels):
    """
    合并已有的标签和新标签，确保所有标签都被正确编码

    :param le:          当前标签解码器
    :param new_labels:  新的标签
    :return:            新的标签解码器
    """
    # 检查编码器是否已拟合（通过判断classes_是否存在）
    if hasattr(le, 'classes_'):
        existing_labels = list(le.classes_)
    else:
        # 编码器未拟合，初始化为空列表
        existing_labels = []

    # 找出新标签中不存在于已有标签中的部分
    new_unique_labels = list(set(new_labels) - set(existing_labels))

    if new_unique_labels:
        logger.info(f"发现新的人脸ID: {', '.join(map(str, new_unique_labels))}，准备将其添加到模型中... ...")
        # 创建新的LabelEncoder并合并所有标签
        new_le = LabelEncoder()
        all_labels = existing_labels + new_unique_labels
        new_le.fit(all_labels)
        return new_le
    else:
        return le


def get_new_label_encoder(face_images, user_id):
    new_features, new_labels = get_feature_and_labels(face_images, user_id)

    if not new_features or not new_labels:
        logger.error("没有有效的训练数据，无法进行训练")
        return False
    # 加载组件
    clf = classifier
    le = label_encoder

    # 检查历史特征和标签文件是否存在
    if os.path.exists(history_features_path) and os.path.exists(history_labels_path):
        train_first = False
        history_features = np.load(history_features_path)
        history_labels = np.load(history_labels_path)
    else:
        # 首次训练时初始化空数组
        train_first = True
        history_features = None
        history_labels = None

    # 增量训练：合并新旧标签
    merged_le = merge_labels(le, new_labels)
    new_encoded = merged_le.transform(new_labels)

    # 合并历史特征和新特征
    if history_features is not None and history_labels is not None:
        # 将历史标签转换为新的编码格式
        history_encoded = merged_le.transform(history_labels)
        all_features = np.vstack([history_features, new_features])
        all_labels = np.hstack([history_encoded, new_encoded])
        logger.info(f"合并历史数据({len(history_features)})和新数据({len(new_features)})")
    else:
        all_features = np.array(new_features)
        all_labels = new_encoded

    le = merged_le

    return clf, le, all_features, all_labels, train_first, new_features


def cv2_train(face_images, user_id, min_acc=0.6):
    """

    :param face_images:     人脸CV图像列表
    :param user_id:         用户对应的id
    :param min_acc:         准确率阀值
    :return:                训练评估成果，True成功，False失败
    """
    start_time = time.time()

    clf, le, all_features, all_labels, train_first, new_features = get_new_label_encoder(face_images, user_id)

    try:
        clf.fit(all_features, all_labels)
        logger.info("【模型样本校验】计算样本准确率，进行模型保存校验... ...")

# 抽取样本进行样本准确率测验------------------------------------------------------------
        face_images_sample = random.sample(face_images, min(10, len(face_images)))
        sample_results = []
        for i in range(len(face_images_sample)):
            res, _ = single_face_image_predict(face_images_sample[i], classifier=clf, label_encoder=le)
            if res is not None:
                sample_results.append(res)

        true_samples = sample_results.count(user_id)
        sample_accuracy = true_samples / len(face_images_sample)
# ----------------------------------------------------------------------------------
        if sample_accuracy >= min_acc or train_first:
            # 保存模型、标签编码器和历史特征
            if train_first:
                logger.info("首次训练，成功录入人脸模型！")
            else:
                logger.info(f"【模型样本校验成功】样本准确率为{sample_accuracy * 100}%，验证通过，保存模型")

            joblib.dump(clf, knn_model_path)
            joblib.dump(le, le_path)
            np.save(history_features_path, all_features)
            np.save(history_labels_path, le.inverse_transform(all_labels))  # 保存原始标签
        else:
            logger.error(f"【模型样本校验失败】样本准确率为{sample_accuracy * 100}%，验证失败，重新录入人脸！")
            return False

        # 计算训练信息
        elapsed_time = time.time() - start_time
        unique_ids = len(le.classes_)

        logger.info(
            f"【训练完成】模型共包含 {unique_ids} 个人脸，"
            f"总样本数: {len(all_features)}，本次新增: {len(new_features)}，"
            f"耗时 {elapsed_time:.2f} 秒。"
        )
        return True

    except Exception as e:
        logger.error(f"训练或保存模型时出错: {e}")
        return False


if __name__ == "__main__":
    import cv2
    from utils.face_utils import img_to_bin

    # test_img = cv2.imread("./facedata/dalin1.jpg")
    # test_img_bin = img_to_bin(test_img)
    # face = face_fetcher(test_img_bin)
    #
    # f = process_single_image(face)
    # print(f)

    # print(f)
    # imgs_bin = [test_img_bin]
    # cv2_train(imgs_bin, user_id=2)
    import pickle
    from face_fetcher import face_fetcher

    with open("./facedata/face_yingrui_bin_data.pkl", "rb") as f:
        data = pickle.load(f)
        face_id = data['face_id']
        images = data['images']

    aface_images = []
    for image in images:
        aface_images.append(face_fetcher(image))

    cv2_train(aface_images, face_id)
