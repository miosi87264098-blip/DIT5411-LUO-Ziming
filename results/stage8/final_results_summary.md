# DIT5411 - Final Machine Learning Results

## Dataset

- Output classes: 13,065
- Original training candidates: 522,127
- Final training originals: 417,719
- Original validation images: 104,408
- Original assignment test images: 162,550
- Classes with test samples: 13,022
- Classes without test samples: 43
- Training items per full epoch: 2,613,000 (417,719 original and 2,195,281 transformed)

## Model

- Model: CNN V2
- Parameters: 978,553
- Input shape: 96 x 96 x 1
- Output logits: 13,065
- Completed training: one coverage epoch and three full epochs
- Final checkpoint: full_13065_v2_full_epoch_003.keras

## Full Training History

| Epoch | Train Acc. | Val Acc. | Train Loss | Val Loss |
|---|---:|---:|---:|---:|
| 1 | 7.95% | 39.95% | 5.5885 | 2.8802 |
| 2 | 28.29% | 64.53% | 3.4261 | 1.5665 |
| 3 | 38.98% | 70.24% | 2.7191 | 1.2931 |

## Filtered Validation Follow-up

- Images: 85,952 from 13,065 classes, after excluding exact matches with development training
- Top-1: 68.06% (58,496 correct)
- Top-5: 89.68% (77,080 correct)
- Macro class accuracy: 68.12%

## Original Assignment Test Results

- Top-1 Accuracy: 67.05%
- Top-5 Accuracy: 88.84%
- Macro Class Accuracy: 67.18%
- Highest original validation accuracy: epoch 3
- Test duration: 26.64 minutes

See the [four-model evaluation](full_class_comparison.json) for the common test run and input hashes.

## Limitations

- The 43 rare categories have no held-out test examples.
- Class-ID to actual Chinese-character mapping still requires verification.
- Three-class MLP/CNN experiments are not equivalent to full-class benchmarking.
- A follow-up content audit found 18,456 validation rows matching development training and 16 white test rows matching development training or validation. See the [test protocol and sensitivity results](../../README.md#7-original-test-results).
- Test results should not be used repeatedly for hyperparameter selection.

## Generated Figures

![Training accuracy](full_training_accuracy.png)

![Training loss](full_training_loss.png)

![Class accuracy distribution](class_accuracy_distribution.png)
