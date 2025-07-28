import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
import torch.optim as optim
from Model import TextClassificationModel, train, evaluate, predict, type_dic, id_to_label, tok
from text_data import text_data
import numpy as np


class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, text_data, label_dict):
        self.tokenizer = tokenizer
        self.data = []

        # 转换文本标签为数字标签
        for text, label_str in text_data:
            label = label_dict.get(label_str, label_dict["其他"])
            self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
num_labels = len(type_dic)
model = TextClassificationModel(num_labels=num_labels)
model.to(device)

# 准备数据
tokenizer = tok
dataset = TextClassificationDataset(tokenizer, text_data, type_dic)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 划分训练集和验证集 (80%训练, 20%验证)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# 设置训练参数
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练和验证
best_val_loss = float('inf')
for epoch in range(10):
    print(f"\nEpoch {epoch + 1}")

    # 训练
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
    print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

    # 验证
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
    print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("保存最佳模型")

# 测试预测
test_texts = [
    "我该上哪里去办理身份证",
    "水费单子字太小看不清",
    "退休金超过五千要交税吗",
    "我想把户口迁到儿子家",
    "这个机器怎么用"
]

print("\n测试预测:")
for text in test_texts:
    pred = predict(model, tokenizer, text, device)
    print(f"文本: '{text}'")
    print(f"预测类别: {pred if pred else '未识别'}")
    print("-" * 50)

# 保存完整模型（包含结构和权重）
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
    'label_map': type_dic
}, "text_classifier_model.pth")