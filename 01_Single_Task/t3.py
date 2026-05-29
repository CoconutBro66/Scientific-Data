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
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import BCEWithLogitsLoss

max_len = 128
batch_size = 16

model_name = "multilingual-e5-base"

train_file = "train_data/train.jsonl"
dev_file   = "train_data/dev.jsonl"
test_file  = "train_data/test.jsonl"

output_dir = "./multilingual-e5-base"

def parse_pos(pos):
    start, end = map(int, pos.split("-"))
    return start, end

def load_loc_dataset(path, tokenizer):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            text = s["text"]
            target = s.get("target","").strip()

            if target:
                combined = f"[TGT] {target} [/TGT] {text}"
            else:
                combined = text

            enc = tokenizer(combined,
                            truncation=True,padding="max_length",
                            max_length=max_len,return_offsets_mapping=True)

            loc = [0.0]*max_len
            for phrase in s["hate_phrases"]:
                start,end = parse_pos(phrase["char_pos"])
                for i,(s1,e1) in enumerate(enc["offset_mapping"]):
                    if e1>start and s1<end:
                        loc[i]=1.0

            data.append({
                "input_ids":enc["input_ids"],
                "attention_mask":enc["attention_mask"],
                "labels":loc
            })
    return Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = load_loc_dataset(train_file,tokenizer)
dev_dataset   = load_loc_dataset(dev_file,tokenizer)
test_dataset  = load_loc_dataset(test_file,tokenizer)

train_dataset.set_format(type="torch")
dev_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")


class LocConfig(PretrainedConfig):
    model_type="loc_model"

class LocModel(PreTrainedModel):
    config_class=LocConfig

    def __init__(self,config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.loc_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size,self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size,1)
        )

    def forward(self,input_ids,attention_mask,labels=None):
        out = self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        hidden = out.last_hidden_state
        
        logits = self.loc_classifier(hidden).squeeze(-1)

        loss=None
        if labels is not None:
            loss = BCEWithLogitsLoss(
                weight=attention_mask.float()
            )(logits,labels.float())

        return {"loss":loss,"logits":logits}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()

    preds = preds.flatten()
    labels = labels.flatten()

    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"loc_f1_macro": f1}


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loc_f1_macro",
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs_loc_baseline_multilingual-e5-base",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    save_total_limit=1
)

if __name__=="__main__":
    model=LocModel(LocConfig())

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("===== 开始 loc 单任务训练 =====")
    trainer.train()

    print("===== 测试集评估 =====")
    test_metrics = trainer.evaluate(test_dataset)

    os.makedirs("results/multilingual-e5-base",exist_ok=True)
    save_path="results/multilingual-e5-base/loc_baseline_results.txt"
    with open(save_path,"w",encoding="utf-8") as f:
        for k,v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    print("结果保存到：", save_path)
    trainer.save_model(output_dir)
    print("模型保存到：", output_dir)
