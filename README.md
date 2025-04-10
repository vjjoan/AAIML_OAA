# AAIML_OAA
## Appliance Classification using Random Forest

This document describes a project focused on classifying household appliances based on their simulated energy consumption patterns using a Random Forest machine learning model.

**(i) Problem Statement:**

The efficient monitoring and management of energy consumption in households can be enhanced by automatically identifying the appliances in use. This project addresses the problem of accurately classifying different household appliances (Fridge, Microwave, TV, Washing Machine, AC) based on their power consumption characteristics and usage duration. Accurate appliance identification can contribute to energy disaggregation, anomaly detection, and personalized energy-saving recommendations.

**(ii) Objectives:**

1.  **Develop a predictive model:** To build a machine learning model capable of accurately classifying different household appliances based on their simulated energy consumption features (mean power, max power, standard deviation of power, and duration of use).
2.  **Optimize model performance:** To identify the optimal hyperparameters for the Random Forest classification algorithm using GridSearchCV to maximize the model's accuracy in classifying the appliances.

**(iii) Methodology Used:**

The following methodology was employed to achieve the project objectives:

1.  **Simulated Data Generation:** A synthetic dataset was created using `numpy` to simulate realistic energy consumption patterns for the five target appliances. The features were designed to exhibit some degree of separability between the appliance classes.
2.  **Data Preprocessing:** The categorical 'appliance' labels were encoded into numerical representations to be compatible with the machine learning model. The features were selected for training. The data was split into training and testing sets to evaluate the model's generalization ability.
3.  **Model Selection:** The Random Forest Classifier, an ensemble learning method known for its robustness and ability to handle complex datasets, was chosen as the classification model.
4.  **Hyperparameter Tuning:** GridSearchCV, a technique for systematically searching through a predefined hyperparameter space, was used to find the best combination of parameters for the Random Forest model. The hyperparameters tuned included `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `class_weight`. Cross-validation was used during the grid search to evaluate the performance of each hyperparameter combination.
5.  **Model Training and Evaluation:** The best model identified by GridSearchCV was trained on the training data. Its performance was then evaluated on the unseen test data using accuracy score and a detailed classification report, which includes precision, recall, and F1-score for each appliance class.

**(iv) Outcomes:**

The primary outcomes of this project are:

1.  **A trained and optimized Random Forest classification model:** This model can predict the type of household appliance based on its simulated energy consumption data. The model's hyperparameters were tuned to achieve high classification accuracy.
2.  **An evaluation of the model's performance:** The accuracy score and classification report on the test data provide insights into the model's ability to generalize to unseen examples and its performance for each specific appliance category. The identified best hyperparameters offer guidance for future model development with similar datasets.
