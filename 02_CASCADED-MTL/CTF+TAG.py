import os
import json
import torch
import numpy as np
from collections import Counter
from datasets import Dataset
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ====================
# 基础参数设置
# ====================
model_name = "/opt/data/private/model/xlm-roberta-base"

train_file = "/opt/data/private/target_detection/Fine-tune/Multi_Task/train_data/train.jsonl"
dev_file = "/opt/data/private/target_detection/Fine-tune/Multi_Task/train_data/dev.jsonl"
test_file = "/opt/data/private/target_detection/Fine-tune/Multi_Task/train_data/test.jsonl"

output_dir = "./finetuned_xlm-roberta-base_multitask_CTF+TAG"
max_len = 128
batch_size = 16

# 标签映射
tri_label2id = {"unidentified-targets": 0, "targeted-abusive": 1, "non-abusive": 2}
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
    "general": 11,
}
fine_id2label = {v: k for k, v in fine_label2id.items()}

num_tri_labels = len(tri_label2id)
num_fine_labels = len(fine_label2id)


# ====================
# 数据加载与预处理
# ====================
def parse_char_pos(char_pos_str):
    if not isinstance(char_pos_str, str) or "-" not in char_pos_str:
        return (0, 0)
    try:
        start, end = map(int, char_pos_str.split("-"))
        return (start, end) if start <= end else (0, 0)
    except:
        return (0, 0)


def load_jsonl_multitask(path, tokenizer):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                if not all(k in sample for k in ["id", "text", "tri_label", "fine_label", "hate_phrases"]):
                    continue

                tri_label_str = sample["tri_label"]
                if tri_label_str not in tri_label2id:
                    continue
                tri_label = tri_label2id[tri_label_str]

                fine_label_str = sample["fine_label"]
                if tri_label == 1 and fine_label_str in fine_label2id:
                    fine_label = fine_label2id[fine_label_str]
                else:
                    fine_label = -1

                text = sample["text"]
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_offsets_mapping=True,
                )

                loc_label = [0.0] * max_len
                for phrase in sample["hate_phrases"]:
                    if not isinstance(phrase, dict) or "char_pos" not in phrase:
                        continue
                    start_char, end_char = parse_char_pos(phrase["char_pos"])
                    for token_idx in range(max_len):
                        token_start, token_end = encoding["offset_mapping"][token_idx]
                        if not (token_end <= start_char or token_start >= end_char):
                            loc_label[token_idx] = 1.0

                data.append(
                    {
                        "id": sample["id"],
                        "text": text,
                        "tri_label": tri_label,
                        "fine_label": fine_label,
                        "loc_label": loc_label,
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                    }
                )
            except Exception as e:
                print(f"解析出错：{e}")
                continue
    print(f"加载 {path} 完成，有效样本数：{len(data)}")
    return Dataset.from_list(data)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = load_jsonl_multitask(train_file, tokenizer)
dev_dataset = load_jsonl_multitask(dev_file, tokenizer)
test_dataset = load_jsonl_multitask(test_file, tokenizer)

for ds in [train_dataset, dev_dataset, test_dataset]:
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "tri_label", "fine_label", "loc_label"])

# 类别权重
train_fine_labels = [x["fine_label"] for x in train_dataset if x["fine_label"] != -1]
label_counts = Counter(train_fine_labels)
total = len(train_fine_labels)
class_weights = torch.tensor([total / label_counts.get(i, 1) for i in range(num_fine_labels)], dtype=torch.float32)


# 模型定义（含 Conditional Fine CTF+TAG）
# ====================
class MultiTaskConfig(PretrainedConfig):
    model_type = "multi_task_model"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 768


class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = config.hidden_size

        # ===== T1: 三分类头 =====
        self.tri_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, num_tri_labels),
        )

        # ===== CTF: tri_logits -> c_tgt 投影 =====
        # c^tgt = W_proj z^tgt
        self.tri_proj = torch.nn.Linear(num_tri_labels, self.hidden_size)

        # ===== TAG: tri_probs -> gate =====
        # g = sigmoid(W_gate p^tgt)
        self.gate_layer = torch.nn.Linear(num_tri_labels, self.hidden_size)

        # ===== T2: 细分类头（输入为 [h_gate ; c_tgt]，维度 2d） =====
        self.fine_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, num_fine_labels)
        )

        # ===== T3: 定位头（不含 LGSD，此处直接用 token 表示） =====
        self.loc_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, 1)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for module in [self.tri_classifier, self.tri_proj, self.gate_layer, self.fine_classifier, self.loc_classifier]:
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tri_label=None,
        fine_label=None,
        loc_label=None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        pooler_output = outputs.last_hidden_state[:, 0, :]          # h_cls
        last_hidden_state = outputs.last_hidden_state  # H

        # ===== T1: tri logits / probs =====
        tri_logits = self.tri_classifier(pooler_output)                 # z^tgt
        tri_probs = torch.softmax(tri_logits, dim=-1)                   # p^tgt

        # ===== TAG: h_gate = h_cls ⊙ sigmoid(W_gate p^tgt) =====
        gate = torch.sigmoid(self.gate_layer(tri_probs))
        h_gate = pooler_output * gate

        # ===== CTF: c_tgt = W_proj z^tgt =====
        tri_proj_vec = self.tri_proj(tri_logits)

        # ===== T2: h_fine = [h_gate ; c_tgt] =====
        fine_input = torch.cat([h_gate, tri_proj_vec], dim=-1)
        fine_logits = self.fine_classifier(fine_input)

        # ===== T3: token-level span logits（不含 LGSD） =====
        loc_logits = self.loc_classifier(last_hidden_state).squeeze(-1)

        # ===== Loss =====
        loss = None
        if tri_label is not None and fine_label is not None and loc_label is not None:
            tri_loss = CrossEntropyLoss()(tri_logits, tri_label)

            # 只在 fine_label!=-1 的样本上计算 T2 loss（数据加载阶段已将非 targeted-abusive 置为 -1）
            fine_mask = fine_label != -1
            if fine_mask.sum() > 0:
                fine_loss = CrossEntropyLoss(weight=class_weights.to(tri_logits.device))(
                    fine_logits[fine_mask], fine_label[fine_mask]
                )
            else:
                fine_loss = torch.tensor(0.0).to(tri_logits.device)

            # T3 默认：所有样本都参与（padding 位置 mask 掉）
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
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="tri_f1_macro",
    logging_dir="./logs_multitask_CTF+TAG_new",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,  # ✅ 只保留最优 checkpoint
)


# ====================
# 评估指标
# ====================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    tri_logits, fine_logits, loc_logits = preds
    tri_labels, fine_labels, loc_labels = labels

    tri_preds = tri_logits.argmax(-1)
    tri_acc = accuracy_score(tri_labels, tri_preds)
    tri_p, tri_r, tri_f1, _ = precision_recall_fscore_support(tri_labels, tri_preds, average="macro", zero_division=0)

    fine_mask = (tri_labels == 1) & (fine_labels != -1)
    fine_f1 = 0.0
    if fine_mask.sum() > 0:
        fine_preds = fine_logits[fine_mask].argmax(-1)
        fine_true = fine_labels[fine_mask]
        _, _, fine_f1, _ = precision_recall_fscore_support(fine_true, fine_preds, average="macro", zero_division=0)

    loc_preds = (torch.sigmoid(torch.tensor(loc_logits)) > 0.5).numpy()
    loc_true = loc_labels
    loc_p, loc_r, loc_f1, _ = precision_recall_fscore_support(loc_true.flatten(), loc_preds.flatten(), average="macro")

    return {"tri_accuracy": tri_acc, "tri_f1_macro": tri_f1, "fine_f1_macro": fine_f1, "loc_f1_macro": loc_f1}


# ====================
# 自定义 Trainer
# ====================
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


# ====================
# 训练与测试
# ====================
if __name__ == "__main__":
    model = MultiTaskModel(MultiTaskConfig())

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("===== 开始多任务训练 (Conditional Fine) =====")
    trainer.train()

    print("===== 测试集评估 =====")
    test_metrics = trainer.evaluate(test_dataset)
    os.makedirs("results/multitask_CTF+TAG", exist_ok=True)
    result_path = f"results/multitask_CTF+TAG/test_metrics_{training_args.num_train_epochs}_epoch_new.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"测试结果已保存到：{result_path}")

    trainer.save_model(output_dir)
    print(f"模型已保存到：{output_dir}")
