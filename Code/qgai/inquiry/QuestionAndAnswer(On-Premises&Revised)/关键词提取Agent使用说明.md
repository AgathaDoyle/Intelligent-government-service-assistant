## 关键词提取Agent

### DsReply

#### 模型与训练结合改进

用Transformer构建关键词提取模型

tokenizer代表分词器

### Model

#### Predict

##### 传入参数

model为要传入的模型（已定义--->EnhancedQAModel）

tokenizer为分词器（已定义--->直接调用BertTokenizerFast.from_pretrained('bert-base-chinese')）

question（我们问的问题 info:str）

context（用户给的回答--->需要与前后端进行交互 info:str）

通过question与context比对，确定关键词

##### 传出参数：

所需要的字符串(info:str)





