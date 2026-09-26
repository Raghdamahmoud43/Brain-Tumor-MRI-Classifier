# Brain Tumor MRI Classification Model — Metadata

## Model Information
- Model name: Brain Tumor MRI Classifier
- Model file: `best_model.keras`
- Model format: Keras
- Architecture: MobileNetV2 (ImageNet pretrained) + custom classification head
- Task: 4-class brain MRI image classification

## Classes
1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

## Input
- Image size: 224 × 224 pixels
- Channels: 3 (RGB)
- Preprocessing: MobileNetV2 `preprocess_input`

## Training Configuration
- Batch size: 32
- Maximum epochs: 15
- Data augmentation:
  - Random horizontal flip
  - Random rotation: 0.08
  - Random zoom: 0.08
- Optimizer: Adam
- Initial learning rate: 0.001
- Loss: Sparse categorical crossentropy
- Metric: Accuracy
- Validation split: 20%
- Random seed: 42
- Early stopping: enabled
- ReduceLROnPlateau: enabled
- Best-model checkpoint: `best_model.keras`

## Dataset Structure
Training and Testing datasets contain four classes:
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Training set: 5,600 images (1,400 per class)  
Testing set: 1,600 images (400 per class)

The training directory was split into 80% training and 20% validation.

## Final Evaluation
- Test accuracy: 84.375%
- Test loss: 0.5114

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Glioma | 0.92 | 0.69 | 0.79 |
| Meningioma | 0.77 | 0.71 | 0.74 |
| No Tumor | 0.90 | 0.98 | 0.94 |
| Pituitary | 0.81 | 0.99 | 0.89 |
| **Overall accuracy** | | | **0.84** |

## Confusion Matrix
```text
[[275  79  24  22]
 [ 22 283  22  73]
 [  1   4 394   1]
 [  2   0   0 398]]
```

## Model File
- File size: 13,590,106 bytes
- Approximate size: 13.6 MB

## Intended Use
This model is intended for educational, research, and portfolio purposes. It is not a medical device and should not be used as a standalone tool for diagnosis, treatment, or clinical decision-making.

## Limitations
Performance can vary on MRI images from different hospitals, scanners, acquisition protocols, patient populations, and preprocessing pipelines. The reported test results apply only to the available test dataset and should not be interpreted as clinical diagnostic performance.
