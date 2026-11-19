# DIT5411-LUO-Ziming

## Traditional Chinese Handwriting Recognition Using TensorFlow

A machine learning project developed for THEi DIT5411 Machine Learning.

This project uses Convolutional Neural Networks (CNNs) to recognize handwritten Traditional Chinese characters from grayscale images.

The final CNN V2 model has 13,065 output classes and includes a Windows desktop interface for single-character recognition.

## 1. Project Overview

- Language: Python 3.12
- Deep learning framework: TensorFlow / Keras
- Computer vision: OpenCV
- Dataset size: 684,677 images
- Character classes: 13,065
- Input shape: 96 x 96 x 1
- Output classes: 13,065
- Final model: CNN V2
- Final checkpoint: full_13065_v2_full_epoch_003.keras

## 2. Dataset

The project uses a Traditional Chinese handwriting dataset containing 684,677 handwritten character images across 13,065 classes.

The dataset is based on handwritten character data originally associated with the Tegaki ecosystem.

Dataset reference:
https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

The original samples are excluded from this Git repository because of their large number and external licensing requirements.

### Data Splitting

For each class, the first 40 available original samples were assigned to the initial training set. Remaining samples were assigned to the independent test set.

Classes with fewer than or equal to 40 samples may have no independent test examples.

Dataset statistics:

| Dataset component | Images |
|---|---:|
| All original samples | 684,677 |
| Initial training samples | 522,127 |
| Independent test samples | 162,550 |
| Development training subset | 417,719 |
| Validation subset | 104,408 |

Validation samples were separated from the initial training set. The independent test set was not used as the validation dataset.

The training and test split was checked for overlapping file paths. A separate content-hash duplicate audit has not been established.

Only 13,022 classes have test images. The other 43 classes have no independent test samples.

## 3. Image Preprocessing

The CNN accepts grayscale images at 96 x 96 pixels.

The inference implementation uses:

1. Grayscale image decoding.
2. Resize to 96 x 96 using OpenCV INTER_AREA.
3. Normalize pixel values to [0, 1].
4. Invert pixel intensity to represent dark handwriting as high-valued strokes.
5. Add the channel and batch dimensions.

The desktop interface reproduces the saved evaluation predictions for 32 out of 32 checked examples. This is a pipeline compatibility check, not a new accuracy measurement.

## 4. Data Augmentation

The assignment requires at least 200 training examples for every class.

The full training pipeline uses 200 training items per class per epoch. These include original images and dynamically augmented versions, producing:

13,065 x 200 = 2,613,000 training items per full epoch.

The augmentation implementation includes geometric transformations such as:

- Small random rotations
- Scaling
- Shearing

The development configuration uses approximately:

- Rotation: +/-8 degrees
- Scaling: 0.90 to 1.08
- Shearing: +/-0.12

Augmentation is applied to training data, not to the independent test images.

Implementation:
`src/augmentation.py`

## 5. Model Development

Several neural network architectures were examined during development.

### Small Three-Class Pilot Experiments

The project includes an initial three-class development experiment with different neural network architectures.

| Pilot architecture | Reported pilot accuracy |
|---|---:|
| MLP | 96.97% |
| CNN V1 | 96.97% |
| Improved CNN V1 | 33.33% |
| Improved CNN V2 | 100.00% |

IMPORTANT: These numbers refer to a small three-class development experiment with 33 evaluation images. They are not performance results for all 13,065 classes.

Related comparison artifacts:
`results/model_comparison_3class_v3.csv`

### Architecture Comparison

The following four models were developed for the three-class pilot task:

| Model | Main architecture | Pilot test accuracy |
|---|---|---:|
| MLP V2 | Flatten, Dense(128), Dropout(0.20), Dense(64), 3-class Softmax | 96.97% |
| CNN V1 | Conv2D(16/32/48), MaxPooling, Flatten, Dense(64), Dropout(0.25), Softmax | 96.97% |
| Improved CNN V1 | Conv2D(16/32/64/96), Batch Normalization, Global Average Pooling, Dense(64), Dropout(0.30), Softmax | 33.33% |
| Improved CNN V2 | Conv2D(16/32/64/96), Layer Normalization, MaxPooling, Flatten, Dense(64), Dropout(0.30), Softmax | 100.00% |

These are pilot results on 33 held-out images from three selected classes, not results on the complete 13,065-class task.

The final full-scale CNN V2 uses a separate architecture defined in `src/full_pilot_train.py`: three convolutional layers with 16, 32 and 48 filters, strided downsampling, pooling, a 64-unit dense layer, dropout of 0.20 and a 13,065-unit logits output. It uses Adam with learning rate 0.0003 and sparse categorical cross-entropy from logits.

The final full-scale CNN V2 should not be confused with the three-class Improved CNN V2 pilot model.

### Full 13,065-Class CNN V2

The final model is a convolutional neural network with multiple feature extraction layers and a 13,065-unit classification output.

The model contains approximately 978,553 parameters.

The final full-scale model was trained for three epochs, with dynamic training augmentation.

Other architecture experiments were conducted during development, but a same-condition full-scale MLP/CNN comparison was not completed. Pilot results must not be directly compared with the final full-scale test results.

## 6. Training Results

| Full epoch | Training accuracy | Validation accuracy | Training loss | Validation loss |
|---|---:|---:|---:|---:|
| 1 | 7.95% | 39.95% | 5.5885 | 2.8802 |
| 2 | 28.29% | 64.53% | 3.4261 | 1.5665 |
| 3 | 38.98% | 70.24% | 2.7191 | 1.2931 |

Training used augmented samples and dropout, while validation used unaugmented held-out images. These factors, together with how metrics are accumulated during an epoch, may contribute to validation accuracy exceeding training accuracy.

### Training Accuracy

![Training accuracy](results/stage8/full_training_accuracy.png)

### Training Loss

![Training loss](results/stage8/full_training_loss.png)

## 7. Independent Test Results

The final checkpoint was evaluated against 162,550 held-out test images.

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 67.05% |
| Top-5 Accuracy | 88.84% |
| Macro Class Accuracy | 67.18% |
| Test images | 162,550 |
| Classes represented in test set | 13,022 |

Macro Class Accuracy is the unweighted mean of the per-class accuracies for classes with test samples. It is NOT macro F1-score.

The three-class pilot results and full-scale test results represent different classification tasks and are not directly comparable.

### Per-Class Accuracy Distribution

![Class accuracy distribution](results/stage8/class_accuracy_distribution.png)

### Recognition Error Analysis

![Error analysis](results/stage8/error_analysis_chinese_labels.png)

The saved visual error examples illustrate characteristic mistakes, but the selected error samples are not a random representative sample of all errors.

## 8. Traditional Chinese Character Mapping

The original ZIP filename bytes were examined to recover character labels.

The recovered mapping contains one candidate character for every class:

| Mapping evidence | Number of classes |
|---|---:|
| Directly consistent encoding | 12,978 |
| Reconstructed filename candidate | 85 |
| Visually supported special candidate | 2 |
| Total | 13,065 |

The mapping is reconstructed and has not been fully checked against an authoritative original label dictionary.

The two special cases are:

- Class 2781: 濈
- Class 12332: 嶒

Their original filenames contain unexplained additional ASCII bytes. These remain visually supported candidates, not independently verified original labels.

Mapping file:
`results/stage9/character_mapping_for_demo.csv`

## 9. Desktop Recognition Demo

A Tkinter desktop application allows users to select a single-character image and display:

- Predicted Traditional Chinese character
- Predicted class ID
- Top-5 character candidates
- Model scores
- Mapping evidence status

### Run the Demo

On Windows with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\handwriting_demo.py
```

The application loads the final CNN V2 checkpoint from `models/` and the character mapping from `results/stage9/`.

### Standalone Demo Compatibility

The public repository includes the final trained checkpoint and the reconstructed class-to-character mapping.

The original training and test images are not distributed.

When the original saved error-sample references are available, the application checks preprocessing compatibility against cached model predictions.

When these references are absent, the application uses the previously validated `area_inverted` preprocessing configuration and skips the dataset-dependent compatibility check.

The input/output model shape and the complete class mapping are still validated.

This fallback does not constitute a new test evaluation and does not change the reported test accuracy.

Users can select their own single-character handwriting images through the Tkinter interface.

### Five Manually Selected Demo Examples

| True character | Top-1 prediction | Model score | True-label rank |
|---|---|---:|---:|
| 不 | 不 | 63.18% | 1 |
| 他 | 佗 | 35.08% | 2 |
| 仙 | 仙 | 99.11% | 1 |
| 佢 | 佢 | 70.52% | 1 |
| 濈 | 激 | 43.99% | 2 |

These five manually selected examples are illustrative functional tests, not a statistically representative evaluation.

## 10. Limitations

1. Recognition accuracy varies substantially between character classes.
2. Only 13,022 of 13,065 classes have independent test samples.
3. Some label mappings required byte reconstruction or visual interpretation.
4. The demo is designed for images containing one handwritten character.
5. Large whitespace, unusual backgrounds and different handwriting styles may reduce performance.
6. Development pilot accuracies cannot be directly compared with the full 13,065-class results.
7. A fully matched full-scale architecture benchmark was not completed.
8. No content-hash audit proving the absence of duplicate image content has been established.

## 11. References

1. THEi, DIT5411 Machine Learning Assignment (Character Recognition), 2026.
2. Traditional Chinese Handwriting Dataset:
   https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
3. OpenCV augmentation reference:
   https://github.com/darshitajain/Introduction-to-OpenCV
4. Keras and TensorFlow example:
   https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow
5. TensorFlow:
   https://www.tensorflow.org/
6. OpenCV:
   https://opencv.org/

## 12. Reproducibility and Data Availability

The repository includes model source code, evaluation summaries, figures and the selected final trained checkpoint.

The full source image dataset is not included. Training and evaluation scripts require the original dataset and locally generated split manifests.

When reusing the original data, check its source attribution and license. The trained model and recovered labels should not be interpreted as an authoritative or unrestricted dataset release.

## Dataset Attribution and Licensing

The handwritten character dataset is credited to AI . FREE Team and Po-Chuan Chen (2020), *Traditional Chinese Handwriting Dataset*.

Source repository:
https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

Complete dataset repository:
https://github.com/chenkenanalytic/handwritting_data_all

The dataset authors state that their work was developed from STUST EECS Chinese MNIST, originally provided by Southern Taiwan University of Science and Technology.

The source dataset is published under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

The original image collection is not redistributed in this repository. The model checkpoint and reconstructed labels originate from work using this dataset; users should check applicable reuse conditions rather than assuming unrestricted redistribution or commercial use.

A separate open-source license for the author's original project code has not yet been selected. Public repository visibility alone does not grant a general software reuse license.

## Author

LUO Ziming

THEi - DIT5411 Machine Learning

Academic Year 2026/27