# Credit_Risk_Analysis

## Overview of the Analysis:

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. We will be using "imbalanced-learn" and "scikit-learn" libraries to build and evaluate models using resampling. We will also be use "RandomOverSampler" and "SMOTE" algorithm to oversample the data, and the "ClusterCentroids" algorithm to undersample the data.


## Results:

```ruby
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = ros_m.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```

### Oversampling
- <I>Naive Random Oversampling:</I> Accuracy Test --> 66.49%; AVG Precision Score --> 99%; AVG Recall Scores --> 60%

    ![Naive Random Oversampling](/Resources/Naive_Random_Oversampling.png) 

- <I>SMOTE Oversampling:</I> Accuracy Test --> 66.23%; AVG Precision Score --> 99%; AVG Recall Scores --> 69%

    ![SMOTE Oversampling](/Resources/SMOTE_Oversampling.png)

### Undersampling

- <I>Undersampling:</I> Accuracy Test --> 54.43%; AVG Precision Score --> 99%; AVG Recall Scores --> 40%

    ![Undersampling](/Resources/Undersampling.png)

### Combination Sampling

- <I>Combination (Over and Under) Sampling:</I> Accuracy Test --> 64.47%; AVG Precision Score --> 99%; AVG Recall Scores --> 57%

    ![Combination Sampling](/Resources/Combination_Sampling.png)

### Ensemble Learners
- <I>Balanced Random Forest Classifier:</I> Accuracy Test --> 78.85%; AVG Precision Score --> 99%; AVG Recall Scores --> 87%

    ![BRF Classifier](/Resources/BRF_Classifier.png)

- <I>Easy Ensemble AdaBoost Classifier:</I> Accuracy Test --> 93.17%; AVG Precision Score --> 99%; AVG Recall Scores --> 94%

    ![EE Classifier](/Resources/EE_Classifier.png)

## Summary:

In the "credit_risk_resampling.ipynb", we tested using Oversampling, Undersampling, and Combination Sampling, which is both (Over and Under). In the "credit_risk_ensemble.ipynb", we tested using Ensemble Learners (Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier). The two Ensemble Learner, Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier, had the two highest accuracy scores and High Risk precision scores. Their accuracy scores are 79% and 93% and High Risk precision scores are 3% and 9% respectively, which are the other four models only had 1%. 

<b><u>Ranking each of the models from most accurate to least:</b></u>

1. Easy Ensemble AdaBoost Classifier
2. Balanced Random Forest Classifier
3. Naive Random Oversampling
4. SMOTE Oversampling
5. Combination (Over and Under) Sampling
6. Undersampling

From the list above, the recommended model to use is the <i>"Easy Ensemble AdaBoost Classifier"</i>, as it has the highest accuracy test results out of the six models we have used. 




