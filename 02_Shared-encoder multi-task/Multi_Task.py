import json
import torch
import os
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from collections import Counter

# ====================
# 参数设置
# ====================
model_name = "multilingual-e5-base"
train_file = "/opt/data/private/target_detection/Fine-tune/Multi_Task/train_data/train.jsonl"
dev_file = "/opt/data/private/target_detection/Fine-tune/Multi_Task/train_data/dev.jsonl"
test_file = "/opt/data/private/target_detection/Fine-tune/Multi_Task/train_data/test.jsonl"
num_tri_labels = 3  # 三分类
num_fine_labels = 12  # 12类细粒度
output_dir = "./finetuned_multilingual-e5-base_multitask"
max_len = 128
batch_size = 16

# 标签映射
tri_label2id = {
    "unidentified-targets": 0,
    "targeted-abusive": 1,
    "non-abusive": 2
}
tri_id2label = {v: k for k, v in tri_label2id.items()}

fine_label2id = {
    "death_threat": 0,
    "sexual_assault": 1,
    "sexual_explicit": 2,
    "physical_harm": 3,
    "radiation_of_threats": 4,
    "attacks_on_credibility": 5,
    "misogynistic": 6,
    "homophobic": 7,
    "religious": 8,
    "political_sectarian": 9,
    "racist": 10,
    "general": 11
}
fine_id2label = {v: k for k, v in fine_label2id.items()}

# ====================
# 工具函数
# ====================
def parse_char_pos(char_pos_str):
    if not isinstance(char_pos_str, str) or "-" not in char_pos_str:
        return (0, 0)
    try:
        start, end = map(int, char_pos_str.split("-"))
        return (start, end) if start <= end else (0, 0)
    except:
        return (0, 0)


# ====================
# 加载数据函数（含 target 融合）
# ====================
def load_jsonl_multitask(path, tokenizer):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                sample = json.loads(line.strip())
                if not all(k in sample for k in ["id", "text", "tri_label", "fine_label", "hate_phrases"]):
                    continue

                # 三分类标签
                tri_label_str = sample["tri_label"]
                if tri_label_str not in tri_label2id:
                    continue
                tri_label = tri_label2id[tri_label_str]

                # 细粒度标签
                fine_label_str = sample.get("fine_label", None)
                if tri_label == 1:
                    if not fine_label_str or fine_label_str not in fine_label2id:
                        continue
                    fine_label = fine_label2id[fine_label_str]
                else:
                    fine_label = -1

                # ====== 核心：将 target 融入输入 ======
                text = sample["text"]
                target = sample.get("target", "").strip()
                if target:
                    combined_text = f"[TGT] {target} [/TGT] {text}"
                else:
                    combined_text = text

                # 编码文本
                encoding = tokenizer(
                    combined_text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_offsets_mapping=True
                )

                # token 级仇恨短语标签
                loc_label = [0.0] * max_len
                hate_phrases = sample.get("hate_phrases", [])
                if isinstance(hate_phrases, list) and len(hate_phrases) > 0:
                    for phrase in hate_phrases:
                        if not isinstance(phrase, dict):
                            continue
                        start_char, end_char = parse_char_pos(phrase.get("char_pos", ""))
                        for token_idx in range(len(encoding["offset_mapping"])):
                            token_start, token_end = encoding["offset_mapping"][token_idx]
                            if not (token_end <= start_char or token_start >= end_char):
                                loc_label[token_idx] = 1.0

                # 加入数据列表
                data.append({
                    "id": sample["id"],
                    "text": combined_text,
                    "tri_label": tri_label,
                    "fine_label": fine_label,
                    "loc_label": loc_label,
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"]
                })
            except Exception as e:
                print(f"[错误] 第{line_idx}行解析失败: {e}")
                continue
    print(f"加载 {path} 完成，有效样本数：{len(data)}")
    return Dataset.from_list(data)


# ====================
# 数据加载
# ====================
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = load_jsonl_multitask(train_file, tokenizer)
dev_dataset = load_jsonl_multitask(dev_file, tokenizer)
test_dataset = load_jsonl_multitask(test_file, tokenizer)

for dataset in [train_dataset, dev_dataset, test_dataset]:
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "tri_label", "fine_label", "loc_label"]
    )

train_fine_labels = [x["fine_label"] for x in train_dataset if x["fine_label"] != -1]
if len(train_fine_labels) == 0:
    raise ValueError("训练集中无有效细分类标签样本！")

label_counts = Counter(train_fine_labels)
total = len(train_fine_labels)
class_weights = torch.tensor(
    [total / label_counts.get(i, 1) for i in range(num_fine_labels)],
    dtype=torch.float32
)


# ====================
# 多任务模型
# ====================
class MultiTaskConfig(PretrainedConfig):
    model_type = "multi_task_model"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = None   # 不写死


class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config):
        super().__init__(config)

        # 换成 XLM-RoBERTa-large
        self.encoder = AutoModel.from_pretrained(model_name)

        # 自动识别 hidden size（非常关键）
        self.hidden_size = self.encoder.config.hidden_size  # XLM-R-large=1024

        # 三个任务的共用分类头
        self.tri_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, num_tri_labels)
        )

        self.fine_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, num_fine_labels)
        )

        self.loc_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.tri_classifier, self.fine_classifier, self.loc_classifier]:
            for layer in module:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(self, input_ids=None, attention_mask=None,
                tri_label=None, fine_label=None, loc_label=None, **kwargs):

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 替换 pooler_output 用 CLS Token（非常关键）
        pooler_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        last_hidden_state = outputs.last_hidden_state       # [batch, seq, hidden]

        # 三分类
        tri_logits = self.tri_classifier(pooler_output)

        # 细分类
        fine_logits = self.fine_classifier(pooler_output)

        # 定位
        loc_logits = self.loc_classifier(last_hidden_state).squeeze(-1)

        loss = None
        if tri_label is not None and fine_label is not None and loc_label is not None:
            tri_loss = CrossEntropyLoss()(tri_logits, tri_label)

            fine_mask = (tri_label == 1) & (fine_label != -1)
            if fine_mask.sum() > 0:
                fine_loss = CrossEntropyLoss(weight=class_weights.to(tri_logits.device))(
                    fine_logits[fine_mask], fine_label[fine_mask]
                )
            else:
                fine_loss = torch.tensor(0.0).to(tri_logits.device)

            loc_loss = BCEWithLogitsLoss(weight=attention_mask.float())(
                loc_logits, loc_label.float()
            )

            loss = 1.2 * tri_loss + 1.0 * fine_loss + 1.0 * loc_loss

        return {
            "loss": loss,
            "tri_logits": tri_logits,
            "fine_logits": fine_logits,
            "loc_logits": loc_logits
        }



# ====================
# 训练参数
# ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",        
    save_strategy="epoch",               
    save_total_limit=1,                 
    load_best_model_at_end=True,         
    metric_for_best_model="tri_f1_macro",
    greater_is_better=True,              
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs_multitask",
    logging_steps=100,
    fp16=torch.cuda.is_available()
)



# ====================
# 指标计算
# ====================
def compute_metrics(eval_pred):
    """计算三分类/细分类/定位任务的指标（兼容旧版transformers输出格式）"""
    outputs = eval_pred.predictions
    labels = eval_pred.label_ids

    # ========== 兼容性判断 ==========
    # 有的版本 Trainer 返回 tuple，有的返回 dict
    if isinstance(outputs, tuple):
        # 按照模型 forward 的返回顺序
        tri_logits, fine_logits, loc_logits = outputs
    elif isinstance(outputs, dict):
        tri_logits = outputs["tri_logits"]
        fine_logits = outputs["fine_logits"]
        loc_logits = outputs["loc_logits"]
    else:
        raise TypeError(f"Unsupported outputs type: {type(outputs)}")

    # ========== 安全访问 inputs ==========
    inputs = getattr(eval_pred, "inputs", None)
    if inputs is not None and "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"].numpy()
    else:
        attention_mask = np.ones_like(labels[2], dtype=np.int32)

    # ========== 三分类任务 ==========
    tri_labels = labels[0]
    tri_preds = tri_logits.argmax(-1)
    tri_acc = accuracy_score(tri_labels, tri_preds)
    tri_p_macro, tri_r_macro, tri_f1_macro, _ = precision_recall_fscore_support(
        tri_labels, tri_preds, average="macro", zero_division=0
    )

    # ========== 细分类任务（仅 targeted-abusive 样本） ==========
    fine_labels = labels[1]
    fine_mask = (tri_labels == 1) & (fine_labels != -1)
    fine_f1_macro = 0.0
    if fine_mask.sum() > 0:
        fine_preds = fine_logits[fine_mask].argmax(-1)
        fine_true = fine_labels[fine_mask]
        _, _, fine_f1_macro, _ = precision_recall_fscore_support(
            fine_true, fine_preds, average="macro", zero_division=0
        )

    # ========== 定位任务 ==========
    loc_labels = labels[2]
    valid_mask = (attention_mask == 1) & (tri_labels[:, None] == 1)
    loc_preds = (torch.sigmoid(torch.tensor(loc_logits)) > 0.5).numpy()[valid_mask]
    loc_true = loc_labels[valid_mask]
    _, _, loc_f1_macro, _ = precision_recall_fscore_support(
        loc_true, loc_preds, average="macro", zero_division=0
    )

    # ========== 汇总 ==========
    return {
        "tri_accuracy": tri_acc,
        "tri_f1_macro": tri_f1_macro,
        "fine_f1_macro": fine_f1_macro,
        "loc_f1_macro": loc_f1_macro
    }




# ====================
# 自定义 Trainer
# ====================
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]



# ====================
# 主程序
# ====================
if __name__ == "__main__":
    config = MultiTaskConfig()
    model = MultiTaskModel(config)

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("===== 开始多任务训练 =====")
    trainer.train()

    print("===== 测试集评估 =====")
    test_metrics = trainer.evaluate(test_dataset)

    os.makedirs("results/MTL_base", exist_ok=True)
    result_path = f"results/MTL_base/test_metrics_{training_args.num_train_epochs}_epoch.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"测试结果已保存到：{result_path}")

    trainer.save_model(output_dir)
    print(f"模型已保存到：{output_dir}")
