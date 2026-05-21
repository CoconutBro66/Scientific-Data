# A Multilingual Dataset of Social Media Abuse Toward Named Public Figures with Layered Evidence
This is the official dataset for the paper **A Multilingual Dataset of Social Media Abuse Toward Named Public Figures with Layered Evidence**.


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
| `target` | The designated Twitter user (named public figure) paired with the post for target-aware abuse annotation. |
| `target_detection_label` | The Target Detection label indicating whether the post is `non-abusive`, `targeted-abusive`, or contains `unidentified-targets`. |
| `abuse_type_label` | The Fine-Grained Abuse Type Classification label assigned to `targeted-abusive` instances, covering one of 12 predefined abuse categories. |
| `evidence_spans` | Character-indexed spans identifying abusive expressions in the original text, formatted as `start-end` (e.g., `25-41`). This field is only available for `targeted-abusive` instances. |




