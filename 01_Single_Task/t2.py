import json
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig
)
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from collections import Counter

# ====================
# 参数
# ====================
# model_name = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Single_Task/model/mdeberta-v3-base"
# model_name = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/public/models/deberta-v3-large"
model_name = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/public/models/xlm-roberta-base"
train_file = "train_data/train.jsonl"
dev_file   = "train_data/dev.jsonl"
test_file  = "train_data/test.jsonl"

max_len = 128
batch_size = 16
output_dir = "./fine_baseline_model_xlm-roberta-base"

fine_label2id = {
    "death_threat": 0,"sexual_assault": 1,"sexual_explicit": 2,"physical_harm": 3,
    "radiation_of_threats": 4,"attacks_on_credibility": 5,"misogynistic": 6,
    "homophobic": 7,"religious": 8,"political_sectarian": 9,"racist": 10,"general": 11
}

num_labels = len(fine_label2id)

# ====================
# 数据加载
# ====================
def load_fine_dataset(path, tokenizer):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            
            # 只用 targeted-abusive 子集
            if s["tri_label"] != "targeted-abusive":
                continue
            if s["fine_label"] not in fine_label2id:
                continue

            text = s["text"]
            target = s.get("target","").strip()
            if target:
                combined = f"[TGT] {target} [/TGT] {text}"
            else:
                combined = text

            enc = tokenizer(combined, truncation=True, padding="max_length", max_length=max_len)

            data.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": fine_label2id[s["fine_label"]]
            })
    return Dataset.from_list(data)


tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = load_fine_dataset(train_file, tokenizer)
dev_dataset   = load_fine_dataset(dev_file, tokenizer)
test_dataset  = load_fine_dataset(test_file, tokenizer)

train_dataset.set_format(type="torch")
dev_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")

# ====================
# 打印标签分布
# ====================
labels = [x["labels"] for x in train_dataset]
cnt = Counter(labels)
total = len(labels)

print("====== Fine-Grained 标签统计（仅 targeted-abusive） ======")
print("训练样本总数 =", total)
print("标签计数 =", cnt)
print("出现过的类别 =", sorted(cnt.keys()))
print("=======================================================")

# ====================
# 安全的 class_weights 计算（不会除 0、不会 NaN）
# ====================
class_weights = []
for i in range(num_labels):
    if cnt[i] == 0:
        # 不存在的类权重设为 1（不影响 loss，不会导致 nan）
        class_weights.append(1.0)
    else:
        class_weights.append(total / cnt[i])

class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("class_weights =", class_weights)


# ====================
# 模型
# ====================
class FineConfig(PretrainedConfig):
    model_type = "fine_model"
    num_labels = num_labels


class FineModel(PreTrainedModel):
    config_class = FineConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0,:]
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weights.to(cls.device))
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


# ====================
# 指标
# ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"fine_f1_macro": macro_f1}


# ====================
# 训练参数
# ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="fine_f1_macro",
    greater_is_better=True,

    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,

    weight_decay=0.01,
    logging_dir="./logs_fine_baseline_xlm-roberta-base",
    logging_steps=50,

    # ❗ 禁用 FP16，以避免 loss = NaN
    fp16=False,

    save_total_limit=1
)

# ====================
# 训练
# ====================
if __name__ == "__main__":
    model = FineModel(FineConfig())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("===== 开始训练 fine-grained 分类（精准） =====")
    trainer.train()

    print("===== 测试集评估 =====")
    test_metrics = trainer.evaluate(test_dataset)

    os.makedirs("results/xlm-roberta-base", exist_ok=True)
    save_path = "results/xlm-roberta-base/fine_baseline_results.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"结果已保存：{save_path}")
    trainer.save_model(output_dir)
    print(f"模型已保存到：{output_dir}")
