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
from torch.nn import CrossEntropyLoss

# ====================
# 参数设置
# ====================
# model_name = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Single_Task/model/mdeberta-v3-base"
# model_name = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/public/models/deberta-v3-large"
model_name = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/public/models/xlm-roberta-base"

train_file = "train_data/train.jsonl"
dev_file   = "train_data/dev.jsonl"
test_file  = "train_data/test.jsonl"

max_len = 128
batch_size = 16
output_dir = "./tri_baseline_model_xlm-roberta-base"

tri_label2id = {
    "unidentified-targets": 0,
    "targeted-abusive": 1,
    "non-abusive": 2
}

# ====================
# 数据加载（与你 MTL 版本一致）
# ====================
def load_tri_dataset(path, tokenizer):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            tri = sample["tri_label"]
            if tri not in tri_label2id:
                continue

            # target 拼接方式保持一致
            text = sample["text"]
            target = sample.get("target", "").strip()
            if target:
                combined_text = f"[TGT] {target} [/TGT] {text}"
            else:
                combined_text = text

            enc = tokenizer(
                combined_text,
                truncation=True,padding="max_length",max_length=max_len
            )

            data.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": tri_label2id[tri]
            })
    return Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = load_tri_dataset(train_file, tokenizer)
dev_dataset   = load_tri_dataset(dev_file, tokenizer)
test_dataset  = load_tri_dataset(test_file, tokenizer)

train_dataset.set_format(type="torch")
dev_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")

# ====================
# 模型定义（单任务）
# ====================
class TriConfig(PretrainedConfig):
    model_type = "tri_model"


class TriModel(PreTrainedModel):
    config_class = TriConfig

    def __init__(self, config):
        super().__init__(config)

        # Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # 单任务头
        self.tri_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, 3)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.tri_classifier(cls)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

# ====================
# 指标
# ====================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(-1)

    acc = accuracy_score(labels, preds)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

    return {
        "tri_accuracy": acc,
        "tri_f1_macro": macro_f1
    }

# ====================
# 训练参数
# ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="tri_f1_macro",
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs_tri_baseline_xlm-roberta-base",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    save_total_limit=1
)

# ====================
# 训练
# ====================
if __name__ == "__main__":
    model = TriModel(TriConfig())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("===== 开始 tri 单任务训练 =====")
    trainer.train()

    print("===== 测试集评估 =====")
    test_metrics = trainer.evaluate(test_dataset)

    os.makedirs("results/xlm-roberta-base", exist_ok=True)
    save_path = "results/xlm-roberta-base/tri_baseline_results.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"结果已保存到：{save_path}")
    trainer.save_model(output_dir)
    print(f"模型已保存到：{output_dir}")
