import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
import tqdm
import math

MAX_LEN   = 256
D_MODEL   = 256
N_HEAD    = 8
N_LAYER   = 4
D_FF      = 1024
VOCAB     = 21128
PAD_ID    = 0

# 修复标签字典映射问题
type_dic={
    "首次办理身份证": 0,
    "户口迁移": 1,
    "水电费缴纳": 2,
    "医保参保": 3,
    "缴纳个人所得税": 4,
    "其他": 5
}

# 添加反向映射，用于预测结果的转换
id_to_label={v: k for k, v in type_dic.items()}

class TextClassificationModel(nn.Module):
    """BERT-based文本分类模型"""
    def __init__(self,num_labels=len(type_dic)):
        super(TextClassificationModel,self).__init__()
        self.bert=BertModel.from_pretrained('bert-base-chinese')
        self.classifier=nn.Linear(self.bert.config.hidden_size,num_labels)
        self.dropout=nn.Dropout(0.3) #添加dropout防止过拟合

    def forward(self,input_ids,attention_mask):
        outputs=self.bert(input_ids,attention_mask=attention_mask)
        pooled_output=outputs.pooler_output
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        return logits


def train(model,dataloader,optimizer,criterion,device):
    model.train()
    total_loss,total_correct,total_samples=0,0,0

    for batch in tqdm.tqdm(dataloader,desc="Training"):
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['label'].to(device)

        optimizer.zero_grad()
        outputs=model(input_ids,attention_mask)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        _, preds=torch.max(outputs, 1)
        total_correct+=(preds == labels).sum().item()
        total_samples+=labels.size(0)

    accuracy=total_correct/total_samples
    return total_loss/len(dataloader),accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return total_loss / len(dataloader), accuracy


def predict(model, tokenizer, text, device, threshold=0.8):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        max_prob, pred_class = torch.max(probs, dim=-1)

    max_prob = max_prob.item()
    pred_class = pred_class.item()

    if max_prob<threshold:
        return None
    return id_to_label[pred_class]

# 初始化tokenizer
tok = BertTokenizerFast.from_pretrained('bert-base-chinese')
