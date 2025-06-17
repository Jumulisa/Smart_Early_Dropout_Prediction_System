# Smart_Early_Dropout_Prediction_System
Smart Early Dropout Prediction System

Project Overview

The objective of this project is to use machine learning techniques to predict student dropout in secondary schools. Accurate predictions can help schools intervene early and reduce dropout rates. The dataset consists of students' academic and demographic information.

I implemented four models:

A basic neural network without any optimization.

An optimized neural network using Adam, L2 regularization, dropout, and early stopping.

A second optimized neural network using RMSprop, L1 regularization, and a higher dropout rate.

An XGBoost classifier with tuned hyperparameters.

The best performing model was the optimized neural network (Model 2).

Dataset

Source: Public student performance dataset

Preprocessing included: one-hot encoding, standardization, label encoding for the target

Target column: Target (with classes 0, 1, 2)

Model Comparison Table

Instance

Model Type

Optimizer

Regularizer

Dropout

Epochs

Early Stopping

Learning Rate

Accuracy

F1 (macro)

1

Neural Net

Adam (default)

None

None

20

 No

Default

36%

0.35

2

Neural Net

Adam

L2

0.3

~20

 Yes

0.001

78%

0.71

3

Neural Net

RMSprop

L1

0.5

~20

 Yes

0.0005

77%

0.68

4

XGBoost

n/a

L1+L2

n/a

150

 (Native)

0.1

78%

0.70

Training Instances (Neural Network Models)

Instance

Optimizer

Regularizer

Epochs

Early Stopping

Layers

Learning Rate

Accuracy

F1 Score

Precision

Recall

1

Adam (default)

None

20

No

3

default

36%

0.35

0.29

0.49

2

Adam

L2

~20

Yes

3

0.001

78%

0.71

0.77

0.70

3

RMSprop

L1

~20

Yes

3

0.0005

77%

0.68

0.75

0.67

4

Adam

L2

~15

Yes

4

0.002

76%

0.69

0.74

0.66

5

Adam

L2

~25

Yes

5

0.0008

78%

0.70

0.76

0.69

Best Performing Model

Model 2: Optimized Neural Network

Outperformed other models with an F1 score of 0.71

Strong recall on underrepresented classes

Generalized well due to combined use of:

Adam Optimizer

L2 Regularization

Dropout (0.3)

EarlyStopping

Tuned learning rate (0.001)

Neural Network vs Classical ML (XGBoost)

Criteria

Neural Network (Model 2)

XGBoost (Model 4)

Accuracy

78%

78%

F1 Score

0.71

0.70

Class 1 Recall

0.40

0.38

Flexibility

High

Medium

Complexity

Higher

Lower

Speed

Slower

Faster

Conclusion: The Neural Network (Model 2) slightly outperformed XGBoost in terms of balanced metrics, especially for minority class 1. It is the recommended model for deployment.

Saved Models

All saved models are found in the /saved_models/ directory:

/saved_models/
    model_1_basic_nn.h5
    model_2_optimized_nn.h5  (BEST MODEL)
    model_3_rmsprop_l1_dropout.h5

How to Run This Project

Open notebook.ipynb in Jupyter or Google Colab

Run all cells from top to bottom

Load the best model for predictions:

from tensorflow.keras.models import load_model
model = load_model('saved_models/model_2_optimized_nn.h5')

Video Presentation (5 minutes)

Includes:

Summary of models

Evaluation metrics

Confusion matrix

Which model worked best and why

Explanation of optimization techniques used

