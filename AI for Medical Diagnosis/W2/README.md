# Week 2 Evaluating Models

Summary: Evaluation metrics for model performance
Status: Done

## Evaluation Metrics

An evaluation metric quantifies how well a predictive model performs. [It involves training the model on a dataset, making predictions on a separate holdout dataset (not used during training), and comparing the predictions to the expected values in the holdout dataset1](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/). Metrics help us understand how accurate and reliable our model’s predictions are.

1. **Classification Metrics**:
    - **Accuracy**: Measures the proportion of correctly predicted instances out of the total instances. When the dataset is imbalanced, accuracy can be misleading. A naive model that predicts all instances as legitimate would achieve 99% accuracy (since it correctly predicts most of the majority class instances). However, this model completely fails to detect fraudulent transactions (the minority class). It’s calculated as:
        
        $\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Instances}}$ 
        
    - **Recall (Sensitivity)**: Measures the proportion of actual positive instances that were correctly predicted. It’s calculated as:
        
        $\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ 
        
    - **Specificity** (also known as the **true negative rate**) is a metric that evaluates a model’s ability to correctly identify instances of the negative class (i.e., the class labeled as “not positive”).
        
        $\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}$
        
    - **Precision (PPV)**: Precision also known as PPV or Positive Predictive Value. It indicates how many of the predicted positive instances are actually positive. It’s calculated as:
        
        $\text{Precision (PPV)} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ 
        
    - **Negative Predictive Value (NPV)** is a statistical metric used in medical tests and diagnostic evaluations. It represents the proportion of true negative results (correctly identified non-disease cases) among all individuals who test negative for a specific condition.
        
        $\text{NPV} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Negatives}}$
        
    - **F1 Score**: Harmonic mean of precision and recall:
        
        $F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ 
        
    - **Cohen’s Kappa**: Measures the agreement between predicted and actual labels beyond chance:
        
        $\kappa = \frac{\text{Observed Agreement} - \text{Expected Agreement}}{1 - \text{Expected Agreement}}$ 
        
    - **ROC Curve**: Graphical representation of the trade-off between true positive rate (sensitivity) and false positive rate.
    - **Area Under the ROC Curve (AUC-ROC)**: Measures the overall performance of a classifier.
2. **Regression Metrics**:
    - **Mean Absolute Error (MAE)**: Average of absolute differences between predicted and actual values.
    - **Mean Squared Error (MSE)**: Average of squared differences between predicted and actual values.
    - **Root Mean Squared Error (RMSE)**: Square root of MSE.
    - **R-squared (Coefficient of Determination)**: Measures the proportion of variance explained by the model.
3. **Clustering Metrics**:
    - Clustering tasks evaluate how well similar items are grouped together. Common metrics include:
        - **Silhouette Score**: Measures the quality of clusters.
        - **Davies-Bouldin Index**: Measures the average similarity between each cluster and its most similar cluster.

## Confusion Metrics

A **confusion matrix** is a powerful tool for assessing the performance of classification models. [It provides a tabular representation of the predicted and actual class labels, enabling us to understand the types of errors made by the model](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)

1. **Components of a Confusion Matrix**:
    - A confusion matrix typically consists of four components:
        - **True Positives (TP)**: Instances correctly predicted as positive (e.g., correctly identifying a disease).
        - **True Negatives (TN)**: Instances correctly predicted as negative (e.g., correctly identifying non-disease cases).
        - **False Positives (FP)**: Instances predicted as positive but are actually negative (e.g., false alarms).
        - **False Negatives (FN)**: Instances predicted as negative but are actually positive (e.g., missing a disease diagnosis).
2. **Use Cases**:
    - Confusion matrices are especially helpful when:
        - Evaluating a model’s performance beyond basic accuracy metrics.
        - Dealing with uneven class distributions in a dataset.
        - Understanding recall, precision, and overall effectiveness in class distinction.
        - Several metrics can be derived from the confusion matrix:
            - **Accuracy**: Ratio of total correct instances to the total instances.
            - **Precision**: Measures how many predicted positive instances are actually positive.
            - **Recall (Sensitivity)**: Proportion of actual positive instances correctly predicted.
            - **Specificity**: Measures true negative rate.
            - **F1 Score**: Harmonic mean of precision and recall.

![Untitled](Week%202%20Evaluating%20Models%20d42208b3f7ca418e96c11929d1bcabaf/Untitled.png)

## ROC Curve

![Untitled](Week%202%20Evaluating%20Models%20d42208b3f7ca418e96c11929d1bcabaf/Untitled%201.png)

1. **What Is the ROC Curve?**
    - The **ROC curve** is a graphical representation that shows the performance of a binary classification model by illustrating the trade-off between two important metrics:
        - **True Positive Rate (TPR)**: Also known as sensitivity or recall. It measures the proportion of actual positive instances correctly predicted by the model.
        - **False Positive Rate (FPR)**: It represents the proportion of actual negative instances incorrectly predicted as positive by the model.
    - The ROC curve plots TPR against FPR at different classification thresholds.
2. **How Does the ROC Curve Work?**
    - Imagine a binary classifier (e.g., disease detection) that produces probability scores for each instance.
    - By varying the classification threshold (the probability value above which an instance is classified as positive), we can calculate TPR and FPR.
    - As we adjust the threshold, the ROC curve traces the trade-off between correctly identifying positives (TPR) and incorrectly flagging negatives (FPR).
3. **Interpretation of the ROC Curve**:
    - The ROC curve visually shows how well the model performs across different thresholds.
    - A model with a curve closer to the top-left corner (higher TPR and lower FPR) is better.
    - The diagonal line (from (0,0) to (1,1)) represents random guessing (no discrimination power).
    - The area under the ROC curve (AUC) quantifies the overall performance:
        - AUC = 1: Perfect classifier.
        - AUC = 0.5: Random guessing.
        - AUC > 0.5: Better than random.
4. **Use Cases and Applications**:
    - **Medical Diagnosis**: Assessing the performance of disease detection models.
    - **Credit Scoring**: Evaluating credit risk models.
    - **Anomaly Detection**: Identifying rare events.
    - **Model Comparison**: Comparing different classifiers.
5. **When to Use the ROC Curve**:
    - Use the ROC curve when:
        - You want to evaluate a binary classification model.
        - You need to choose an optimal threshold based on the trade-off between TPR and FPR.
        - You want to visualize the model’s performance.

## Probability Calibration

**[Probability calibration** is a technique used in machine learning to adjust the predicted probabilities of a classification model so that they better represent the true likelihood of an event occurring](https://medium.com/datainc/probability-caliberation-on-imbalanced-data-792e3add4efa)

Imagine you have a weather forecasting model that predicts the probability of rain tomorrow. If the model says there's an 80% chance of rain, you'd likely take an umbrella when you leave the house. However, if, over time, you notice that it only actually rains 30% of the times the model predicts 80%, you'd lose trust in the model's predictions. Probability calibration aims to align these predicted probabilities with the real-world frequencies, so an 80% prediction means it rains 8 out of 10 times you receive such a forecast.

1. **Predicting Probabilities**:
    - In a classification problem, instead of directly predicting class labels, a model may predict the **probability** of an observation belonging to each possible class label. This approach provides flexibility in interpretation, presentation (choice of threshold and prediction uncertainty), and model evaluation.
    - However, not all machine learning models produce well-calibrated probabilities.  [Complex nonlinear algorithms, especially those that don’t directly make probabilistic predictions, may exhibit discrepancies between predicted probabilities and observed probabilities in the training data2](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/).
    - The goal is to adjust the distribution of predicted probabilities to better match the expected distribution observed in the data.
2. **Calibration of Predictions**:
    - **Reliability diagrams** are commonly used to diagnose the calibration of a model. These diagrams compare predicted probabilities against the actual observed probabilities for different bins or intervals.
    - Methods can be applied to better calibrate predictions, ensuring that the predicted probabilities align with the true probabilities of class membership.
    - [Well-calibrated classifiers allow the output of the **`predict_proba`** method to be directly interpreted as a confidence level3](https://scikit-learn.org/stable/modules/calibration.html).
    
    **Why Is Probability Calibration Important?**
    
    Imagine you have a weather forecasting model that predicts the probability of rain tomorrow. If the model says there's an 80% chance of rain, you'd likely take an umbrella when you leave the house. However, if, over time, you notice that it only actually rains 30% of the times the model predicts 80%, you'd lose trust in the model's predictions. Probability calibration aims to align these predicted probabilities with the real-world frequencies, so an 80% prediction means it rains 8 out of 10 times you receive such a forecast.
    
    **How It Works**
    
    Probability calibration involves applying a transformation to the output of a classification model to ensure that the predicted probabilities match the observed frequencies. Two common methods for probability calibration are:
    
    - Platt Scaling (Logistic Calibration): This method fits a logistic regression model to the raw predictions of your classifier, effectively squashing the output to a probability scale.
    - Isotonic Regression: This is a non-parametric approach that fits a piecewise constant non-decreasing function to the raw model outputs. It's more flexible than Platt scaling but can be prone to overfitting with small datasets.