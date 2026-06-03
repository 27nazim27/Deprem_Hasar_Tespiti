# Earthquake Building Damage Assessment Using Deep Learning and UAV Imagery

## Overview

Rapid and accurate assessment of post-earthquake building damage is essential for effective disaster response and rescue operations. Traditional field inspections are often time-consuming, labor-intensive, and costly. Recent advances in Unmanned Aerial Vehicles (UAVs) and Deep Learning provide a powerful alternative for automated damage assessment.

This project presents a comparative study of a custom Convolutional Neural Network (CNN) and a transfer learning-based MobileNetV2 model for post-earthquake building damage classification using UAV imagery.

---

## Dataset

The experiments were conducted using the **UAVs-based Turkey Earthquake Building Damage Estimation (UAVs-TEBDE)** dataset.

### Damage Classes

* Collapsed
* Damaged
* Intact

The dataset contains UAV-captured building images collected after earthquake events and categorized according to their damage levels.

---

## Objectives

* Develop an automated earthquake damage assessment system.
* Compare the performance of CNN and MobileNetV2 architectures.
* Evaluate the effectiveness of transfer learning for disaster management applications.
* Support rapid and reliable post-disaster decision-making.

---

## Models

### Custom CNN

A Convolutional Neural Network was designed and trained from scratch for multi-class building damage classification.

### MobileNetV2

A transfer learning approach based on MobileNetV2 was implemented to improve classification performance while reducing computational complexity.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Experimental Results

| Model       | Validation Accuracy |
| ----------- | ------------------- |
| CNN         | 82.58%              |
| MobileNetV2 | 95.76%              |

### Key Findings

* MobileNetV2 achieved significantly higher classification accuracy.
* MobileNetV2 produced lower validation loss values.
* Transfer learning reduced classification errors.
* MobileNetV2 demonstrated superior generalization performance on unseen data.

The results indicate that transfer learning can substantially improve post-earthquake building damage assessment systems based on UAV imagery.

---

## Performance Visualizations

### Training and Validation Accuracy

(Add your accuracy graph here)

```markdown
![Accuracy Curve](results/accuracy_curve.png)
```

### Training and Validation Loss

(Add your loss graph here)

```markdown
![Loss Curve](results/loss_curve.png)
```

### Confusion Matrix

(Add your confusion matrix here)

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```

---

## Project Structure

```text
Earthquake_Damage_Detection/
│
├── dataset/
├── models/
├── notebooks/
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
│
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Future Work

* Real-time UAV image processing
* Multi-level damage severity estimation
* Web-based deployment
* Mobile application integration
* Integration with emergency response systems

---

## Conclusion

This study demonstrates that transfer learning-based MobileNetV2 significantly outperforms a custom CNN model in post-earthquake building damage classification tasks. The proposed approach can contribute to faster and more reliable disaster assessment processes, supporting emergency response teams during critical situations.

---

## Author

Nazım Çelebi

Bitlis Eren University
Department of Computer Engineering
