# Gesture Recognition Project
# Gesture Recognition with Explainable AI for Autonomous Systems

## Project Overview

This project presents a hand gesture recognition system for autonomous and intelligent systems using deep learning and Explainable Artificial Intelligence (XAI). The model recognizes ten different hand gestures and converts them into autonomous commands such as STOP, MOVE FORWARD, TURN LEFT, TURN RIGHT, and EMERGENCY HALT.

The project compares MobileNetV2 and EfficientNetB0 transfer learning models and incorporates multiple explainability techniques including Grad-CAM, LIME, and Integrated Gradients to improve model transparency and trustworthiness.


## Features

* Hand Gesture Recognition using Deep Learning
* MobileNetV2 Transfer Learning with Fine-Tuning
* EfficientNetB0 Transfer Learning
* Explainable AI (XAI)

  * Grad-CAM
  * LIME
  * Integrated Gradients
* TensorFlow Lite Edge Deployment
* Autonomous Command Generation
* Performance Evaluation and Model Comparison


## Dataset

Dataset: LeapGestRecog

Number of Classes: 10

Gesture Classes:

1. Palm → STOP
2. L → TURN LEFT
3. Fist → EMERGENCY HALT
4. Fist Moved → MOVE FORWARD
5. Thumb → CONFIRM
6. Index → TURN RIGHT
7. OK → TASK COMPLETE
8. Palm Moved → SLOW DOWN
9. C → SCAN AREA
10. Down → DESCEND


## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* LIME
* TensorFlow Lite


## Model Architecture

### MobileNetV2

* Pretrained on ImageNet
* Feature Extraction
* Fine-Tuning of top layers
* Lightweight and edge-device friendly

### EfficientNetB0

* Pretrained on ImageNet
* Efficient scaling strategy
* High classification performance


## Explainable AI Techniques

### Grad-CAM

Highlights image regions that influence model decisions.

### LIME

Provides local explanations for individual predictions.

### Integrated Gradients

Measures feature importance using attribution scores.



## Performance Evaluation

Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Edge Deployment

The trained MobileNetV2 model is converted to TensorFlow Lite format for deployment on embedded and edge devices.

Outputs:

* mobilenet_gesture.tflite
* Latency Analysis
* FPS Measurement


## Results

Generated Outputs:

* dataset_samples.png
* training_results.png
* confusion_matrix.png
* gradcam_xai.png
* lime_xai.png
* ig_mobilenet.png
* xai_metrics_table.png


