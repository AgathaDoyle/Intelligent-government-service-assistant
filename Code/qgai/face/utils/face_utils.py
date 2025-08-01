# ================================
# @File         : face_utils.py
# @Time         : 2025/07/22
# @Author       : Yingrui Chen
# @description  : 人脸处理公共工具类
# ================================

import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import logging


def bin_to_image_array(bin_data):
    """
    将二进制图像数据转换为三维矩阵
    """
    try:
        image_stream = BytesIO(bin_data)
        img = Image.open(image_stream)
        img = np.array(img)
        return img.copy()
    except Exception as e:
        logging.error(f"图像转换失败: {str(e)}")
        return None


def img_to_bin(image):
    """
    将OpenCV格式图像转为二进制数据
    :param image: OpenCV图像格式(numpy array)，BGR通道
    :return: 图像二进制数据
    """
    # 检查输入是否为有效的OpenCV图像
    if not isinstance(image, np.ndarray) or len(image.shape) not in (2, 3):
        raise ValueError("输入必须是OpenCV格式的图像(numpy数组)")

    ret, buf = cv2.imencode('.jpg', image)
    if not ret:
        raise RuntimeError("无法将图像编码为JPEG格式")

    # 将numpy数组转换为字节流
    return buf.tobytes()


def cv_img_to_bin(image):
    """
    将cv捕捉的图像转成二进制
    :param image: OpenCV读取的图像（numpy数组格式）
    :return: 图像的二进制数据，如果转换失败则返回None
    """
    try:
        # 检查输入是否为有效的图像
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("输入不是有效的OpenCV图像")

        # 将图像编码为JPEG格式，返回值为(retval, buf)
        # retval为布尔值，表示编码是否成功
        # buf为包含编码后数据的字节流
        retval, buffer = cv2.imencode('.jpg', image)

        if not retval:
            raise RuntimeError("图像编码失败")

        # 将numpy数组转换为二进制数据
        binary_data = buffer.tobytes()

        return binary_data
    except Exception as e:
        print(f"图像转换为二进制时发生错误: {str(e)}")
        return None


def extract_features(face_image, embedder):
    """
    使用预训练模型提取人脸特征向量
    """
    # 构建输入 blob
    # face_blob = cv2.dnn.blobFromImage(
    #     face_img, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False
    # )
    feature = embedder.get_feat(face_image)
    feature = np.array(feature, dtype=np.float32).flatten()

    return feature


def normalize_features(features):
    features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    return features


if __name__ == "__main__":
    pass
