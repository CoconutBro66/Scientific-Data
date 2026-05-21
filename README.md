# A Multilingual Dataset of Social Media Abuse Toward Named Public Figures with Layered Evidence
This is the official code and dataset for the paper **A Multilingual Dataset of Social Media Abuse Toward Named Public Figures with Layered Evidence**.


## Dataset
The `data` folder contains the datasets used in our experiment, with split into 8:1:1 ratio. <br>

- **Training Set:** `data/train.jsonl` <br>
- **Dev Set:** `data/dev.jsonl` <br>
- **Test Set:** `data/test.jsonl` <br>

*Note: 'no-target.jsonl' contains posts without a specified target and is not used for training.* <br>

### Dataset description

| Field | Description |
|---|---|
| `id` | The unique identifier of the tweet (Tweet ID). |
| `target` | The designated Twitter user (public figure) that this instance is paired with. |
| `target_detection_label` | The Task 1 (T1) tri-class label: `non-abusive`, `targeted-abusive`, or `unidentified-targets`. |
| `abuse_ type_label` | The Task 2 (T2) fine-grained abuse type label (only applicable when `tri_label` is `targeted-abusive`). |
| `evidence_spans` |  Character-level span position of an abusive phrase in the original text, formatted as `start-end` (e.g., `25-41`). |




