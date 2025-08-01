# ================================
# @File         : Predict.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 用于人脸预测，主函数：cv2_predict()
#                 输入一组二进制图像列表、置信度阀值
#                 输出预测的用户ID
# ================================

import time
from multiprocessing import Pool
from os import cpu_count
import numpy as np
from model_loader import get_components
from utils.face_utils import (
    extract_features, normalize_features
)

# 获取组件
components = get_components()
logger = components['logger']
paths = components['paths']
models = components['models']

# 加载特征提取模型、检测器等
embedder = models['embedder']
detectors = models['detectors']
rec_model = models['rec_model']
clf = models['knn_classifier']
le = models['label_encoder']

DISTANCE_THRESHOLD = 3


def single_face_image_predict(face_image, min_acc=0.96, classifier=clf, label_encoder=le):
    """
    单张照片预测函数，返回两个参数
    预测成功则返回用户ID、对应的置信度
    预测失败则返回两个None

    :param face_image:      人脸CV图像
    :param min_acc:         置信度阀值，低于阀值的数据将不被认可
    :param classifier:      分类器（默认为贝叶斯分类器）
    :param label_encoder:   标签解码器
    :return:                用户ID，置信度
    """
    # 使用公共人脸检测方法
    features = extract_features(face_image, rec_model)
    features = normalize_features(features)
    features = features.reshape(1, -1)

    # 识别人脸
    try:
        distances, _ = classifier.kneighbors(features)
        min_distance = distances[0][0]
        if min_distance < DISTANCE_THRESHOLD:
            predictions = classifier.predict_proba(features)
        else:
            logger.warning(f"未录入人脸，请录入后再识别！KNN最小距离{min_distance:.4f} >= {DISTANCE_THRESHOLD}")
            return None, None

        max_prob = np.max(predictions)
        predicted_class = np.argmax(predictions)

        # 转换为原始标签
        username = label_encoder.inverse_transform([predicted_class])[0]
        confidence_text = f"{max_prob * 100}"

        # 应用置信度阈值
        if max_prob < min_acc:
            logger.info(
                f"【筛选无效数据】{str(username)}，置信度{f'{max_prob * 100:.2f}%'} < {min_acc * 100:.2f}%"
            )
            return None, None

        return username, confidence_text
    except Exception as e:
        logger.error(f"【预测过程出错】: {e}")
        return None, None


def predict_helper(args):
    """多张照片并行处理辅助函数"""
    return single_face_image_predict(*args)


def cv2_predict(face_images, min_acc=0.96, single_pool_mode=True):
    """
    批量处理图像并返回预测结果

    :param single_pool_mode:    指定是否为单线程模式，对于数量少的图片建议使用，默认开启
    :param face_images:         一组CV人脸图片列表
    :param min_acc:             置信度阀值，低于阀值的数据将不被认可
    :return:                    预测的用户ID
    """

    # 初始化标签-置信度列表字典
    logger.info(f"检测到{len(face_images)}个图像数据，开始筛选置信度高于阀值{min_acc}的有效人脸数据... ...")
    start_time = time.time()
    label_confidences = {}
    valid_results = []

    arg_list = [(face_image, min_acc) for face_image in face_images]

    if single_pool_mode:
        # 单线程模式
        for args in arg_list:
            # 直接调用处理函数，无需通过进程池
            username, confidence_text = predict_helper(args)
            if username is not None and confidence_text is not None:
                # 转换类型并添加到有效结果列表
                username_str = str(username)
                confidence = float(confidence_text)
                valid_results.append((username_str, confidence))

                # 同时更新标签-置信度字典
                if username_str not in label_confidences:
                    label_confidences[username_str] = []
                label_confidences[username_str].append(confidence)
    else:
        # 多线程模式
        num_processes = min(cpu_count(), len(face_images))
        with Pool(num_processes) as pool:
            for username, confidence_text in pool.imap(predict_helper, arg_list):
                if username is not None and confidence_text is not None:
                    # 转换类型并添加到有效结果列表
                    username_str = str(username)
                    confidence = float(confidence_text)
                    valid_results.append((username_str, confidence))

                    # 同时更新标签-置信度字典（如果需要）
                    if username_str not in label_confidences:
                        label_confidences[username_str] = []
                    label_confidences[username_str].append(confidence)

    end_time = time.time()
    logger.info(f"筛选出 {len(valid_results)} 个有效人脸数据，用时{end_time-start_time:.4f}s，开始预测")

    # 筛选后没有人脸
    if len(valid_results) == 0:
        return None

    # 构建标签到置信度列表的映射
    for label, conf in valid_results:
        if label in label_confidences:
            label_confidences[label].append(conf)
        else:
            label_confidences[label] = [conf]

    # 计算其他统计信息
    count = {label: len(conf_list) for label, conf_list in label_confidences.items()}       # 每个标签预测的数量字典
    best_label = max(count, key=count.get)                                                  # 数量最多的标签
    best_confidence = label_confidences[best_label]                                         # 数量最多的标签对应的置信度列表
    best_confidence_avg = np.mean(best_confidence)                                          # 置信度列表求平均值

    logger.info(f"【预测结果】用户ID：{best_label} ，置信度：{best_confidence_avg:.4f}%")

    if best_confidence_avg < min_acc:
        return None

    return best_label


if __name__ == "__main__":
    # pass
    # # 单张照片测试
    import cv2
    from face_fetcher import face_fetcher
    from utils.face_utils import img_to_bin

    test_img = cv2.imread("./facedata/gou.jpg")        # 读取图像
    test_img = img_to_bin(test_img)                         # 转二进制
    face_image = face_fetcher(test_img)                     # 获取人脸CV图像
    name, conf = single_face_image_predict(face_image, min_acc=0.96)  # 传入函数

    print(f"预测为{name}, 置信度为{conf}%")

    # print(conf)

    # img_bin = [img_to_bin(test_img)]
    # print(cv2_predict(img_bin, min_acc=0.5))
    # start_time = time.time()
    # res, _ = single_img_bin_predict(img_bin, min_acc=0.1)
    # end_time = time.time()
    # print(res, end_time - start_time)

    # # 照片组
    # import pickle
    # from face_fetcher import face_fetcher
    #
    # with open("./facedata/face_yingrui_bin_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     face_id = data['face_id']
    #     images = data['images']
    #
    # a_face_images = []
    # for image in images:
    #     a_face_images.append(face_fetcher(image))
    #
    # print(cv2_predict(a_face_images, min_acc=0.96, single_pool_mode=True))
