# ================================
# @File         : model_loader.py
# @Time         : 2025/07/24
# @Author       : Yingrui Chen
# @description  : 人脸贝叶斯模型初始化器，负责配置路径、日志和加载基础模型组件
# ================================

import logging
import os
import time
import cv2
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FaceModelLoader:
    """人脸贝叶斯模型初始化器，负责配置路径、日志和加载基础模型组件"""

    def __init__(self):
        # 初始化路径
        self.trainer_dir = 'face_models'
        self.face_data_path = 'facedata'
        self.nn_model_path = "face_models/nn4.small2.v1.t7"
        self.insightface_model_path = "face_models/face_fetcher.pkl"

        # 模型和数据文件路径
        self.knn_model_path = os.path.join(self.trainer_dir, 'knn_model.pkl')
        self.le_path = os.path.join(self.trainer_dir, 'label_encoder.pkl')
        self.history_features_path = os.path.join(self.trainer_dir, 'history_features.npy')
        self.history_labels_path = os.path.join(self.trainer_dir, 'history_labels.npy')

        # 模型组件
        self.embedder = None
        self.rec_model = None
        self.detectors = []

        # 初始化工作流
        self._setup_directories()
        # self._validate_data_path()
        self._load_rec_model()
        self._load_embedder()
        self._load_detectors()

        # 初始化空模型（如果模型文件不存在）
        self.knn_clf = self._initialize_knn_model()
        self.le = self._initialize_label_encoder()

    def _setup_directories(self):
        """确保训练结果目录存在"""
        try:
            os.makedirs(self.trainer_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"创建训练目录失败: {e}")
            raise

    def _validate_data_path(self):
        """验证人脸数据路径是否有效"""
        if not os.path.exists(self.face_data_path):
            logger.error(f"人脸数据路径不存在: {self.face_data_path}")
            raise FileNotFoundError(f"人脸数据路径不存在: {self.face_data_path}")

        if not os.path.isdir(self.face_data_path):
            logger.error(f"人脸数据路径不是一个目录: {self.face_data_path}")
            raise NotADirectoryError(f"人脸数据路径不是一个目录: {self.face_data_path}")

    def _load_rec_model(self):
        """加载预训练的人脸特征提取模型（InsightFace）"""
        try:
            if not os.path.exists(self.insightface_model_path):
                logger.error("InsightFace特征提取模型不存在:{self.insightface_model_path}")
                raise FileNotFoundError(self.insightface_model_path)

            self.rec_model = joblib.load(self.insightface_model_path)

        except Exception as e:
            logger.error(f"加载特征提取模型失败: {e}")
            raise

    def _load_embedder(self):
        """加载预训练的人脸特征提取模型（OpenFace）"""
        try:
            if not os.path.exists(self.nn_model_path):
                logger.error(f"特征提取模型文件不存在: {self.nn_model_path}")
                logger.info("请从 https://cmusatyalab.github.io/openface/models-and-accuracies/ 下载模型")
                raise FileNotFoundError(f"特征提取模型文件不存在: {self.nn_model_path}")

            self.embedder = cv2.dnn.readNetFromTorch(self.nn_model_path)
        except Exception as e:
            logger.error(f"加载特征提取模型失败: {e}")
            raise

    def _load_detectors(self):
        """加载多个检测器以提高检测鲁棒性"""
        detector_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        ]

        for detector_path in detector_paths:
            if not os.path.exists(detector_path):
                logger.warning(f"检测器文件不存在: {detector_path}")
                continue

            detector = cv2.CascadeClassifier(detector_path)
            if not detector.empty():
                self.detectors.append(detector)
            else:
                logger.warning(f"无法加载检测器: {detector_path}")

        if not self.detectors:
            logger.error("无法加载任何人脸检测器")
            raise RuntimeError("无法加载任何人脸检测器")

    def _initialize_knn_model(self):
        """初始化KNN分类模型，如果已有模型则加载"""
        try:
            if os.path.exists(self.knn_model_path):
                import joblib
                knn_clf = joblib.load(self.knn_model_path)
                return knn_clf
            else:
                return KNeighborsClassifier()
        except Exception as e:
            logger.error("初始化KNN模型失败：{e}")
            raise

    def _initialize_label_encoder(self):
        """初始化标签编码器，如果已有则加载"""
        try:
            if os.path.exists(self.le_path):
                # 这里假设使用joblib保存和加载编码器
                import joblib
                le = joblib.load(self.le_path)
                return le
            else:
                return LabelEncoder()
        except Exception as e:
            logger.error(f"初始化标签编码器失败: {e}")
            raise

    def get_components(self):
        """获取所有初始化的组件"""
        return {
            'logger': logger,
            'paths': {
                'trainer_dir': self.trainer_dir,
                'knn_model_path': self.knn_model_path,
                'le_path': self.le_path,
                'history_features_path': self.history_features_path,
                'history_labels_path': self.history_labels_path
            },
            'models': {
                'rec_model': self.rec_model,
                'embedder': self.embedder,
                'detectors': self.detectors,
                'knn_classifier': self.knn_clf,
                'label_encoder': self.le
            }
        }


# 全局单例组件，确保只初始化一次
_global_loader = None
_global_components = None


def get_components():
    global _global_loader, _global_components
    if _global_components is None:
        start_time = time.time()
        _global_loader = FaceModelLoader()
        _global_components = _global_loader.get_components()
        end_time = time.time()
        logger.info(f"人脸识别识别Agent组件加载完毕！共耗时{end_time - start_time:.4f}s")
    return _global_components


if __name__ == '__main__':
    components = get_components()
    for k, v in components.items():
        print(k, v)
