import torch
from transformers import BertTokenizerFast
from Model import TextClassificationModel, id_to_label
import os


class TextClassifier:
    def __init__(self, model_path="text_classifier_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载保存的模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.tokenizer = checkpoint['tokenizer']
        self.label_map = checkpoint['label_map']
        self.id_to_label = {v: k for k, v in self.label_map.items()}

        # 初始化模型
        self.model = TextClassificationModel(num_labels=len(self.label_map))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def classify(self, text, threshold=0.8):
        if not text.strip():
            raise ValueError("输入文本不能为空")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            max_prob, pred_class = torch.max(probs, dim=-1)

        max_prob = max_prob.item()
        pred_class = pred_class.item()

        if max_prob < threshold:
            return None
        return self.id_to_label[pred_class]


# 使用示例
if __name__ == "__main__":
    classifier = TextClassifier()

    test_texts = [
        "我还没有办理过身份证，要怎么弄啊这",
        "水费单子字太小看不清",
        "退休金超过五千要交税吗",
        "我想把户口迁到儿子家",
        "这个机器怎么用"
    ]

    for text in test_texts:
        result = classifier.classify(text)
        print(f"文本: '{text}'")
        print(f"分类结果: {result if result else '未识别'}")
        print("-" * 50)