# Smart_Early_Dropout_Prediction_System

##  Overview

In many developing regions, student dropout rates in secondary schools are alarmingly high, often due to socioeconomic factors, academic struggles, and limited access to educational support. This project leverages machine learning models to predict students at risk of dropping out, enabling timely interventions. The dataset includes academic records and demographic attributes of students, with the `Target` variable representing dropout categories.

This is the link to the DataSet that I have used " https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success?utm_source=chatgpt.com "

This is the link uploaded to youtube explaining the project and all the models " https://www.youtube.com/watch?v=fiQtkMzTavI "

---

##  Objective

To build and evaluate multiple machine learning models that can classify students based on their risk of dropout using different optimization and regularization techniques.

---

##  Training Instances Table

| Instance | Model Type | Optimizer      | Regularizer | Dropout | Epochs | Early Stopping | Learning Rate | Accuracy | F1 (macro) |
| -------- | ---------- | -------------- | ----------- | ------- | ------ | -------------- | ------------- | -------- | ---------- |
| **1**    | Neural Net | Adam (default) | None        | None    | 20     |  No           | Default       | 36%      | 0.35       |
| **2**    | Neural Net | Adam           | L2          | 0.3     | \~20   |  Yes          | 0.001         | **78%**  | **0.71**   |
| **3**    | Neural Net | RMSprop        | L1          | 0.5     | \~20   |  Yes          | 0.0005        | 77%      | 0.68       |
| **4**    | XGBoost    | n/a            | L1+L2       | n/a     | 150    |  (Native)     | 0.1           | **78%**  | 0.70       |

---

##  Summary of Results

* **Best Performing Model**: Instance 2 (Neural Network with Adam, L2 regularization, dropout, and early stopping)
* It achieved the highest macro F1-score (0.71), showing strong balance across all dropout classes.
* Compared to XGBoost (Instance 4), the optimized neural network handled class imbalance better.
* Model 1 performed poorly due to lack of optimization.

---

##  Saved Models Directory

Placed all trained models in the following structure:

```
/saved_models/
├── model_1_basic_nn.h5
├── model_2_optimized_nn.h5  #Best Model
├── model_3_rmsprop_l1_dropout.h5
```

---

##  Run Instructions

1. Open `notebook.ipynb`
2. Run all cells to train or evaluate models
3. Load the best model:

```python
from tensorflow.keras.models import load_model
model = load_model('saved_models/model_2_optimized_nn.h5')
```

---

##  Video Presentation Checklist

* Brief introduction of the project and dataset
* The table of training instances
* Summary of results and best model
* Explanation of model choices and hyperparameters
* Visuals: confusion matrix and plots (if available)
