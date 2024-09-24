## AnomaData(Automated Anomaly Detection for Predictive Maintenance)

### Problem Statement:
Many different industries need predictive maintenance solutions to reduce risks and gain actionable insights through processing data from their equipment.

Although system failure is a very general issue that can occur in any machine, predicting the failure and taking steps to prevent such failure is most important for any machine or software application.

Predictive maintenance evaluates the condition of equipment by performing online monitoring. The goal is to perform maintenance before the equipment degrades or breaks down.
This Capstone project is aimed at predicting the machine breakdown by identifying the anomalies in the data.

The data we have contains about 18000+ rows collected over few days. The column ‘y’ contains the binary labels, with 1 denoting there is an anomaly. The rest of the columns are predictors. 

The following is recommendation of the steps that should be employed towards attempting to solve this problem statement:

**Exploratory Data Analysis:** Analyze and understand the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations.

**Data Cleaning:** This might include standardization, handling the missing values and outliers in the data.

**Feature Engineering:** Create new features or transform the existing features for better performance of the ML Models.

**Model Selection:** Choose the most appropriate model that can be used for this project.

**Model Training:** Split the data into train & test sets and use the train set to estimate the best model parameters.

**Model Validation:** Evaluate the performance of the model on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting.

**Model Deployment:** Model deployment is the process of making a trained machine learning model available for use in a production environment.

##### Exploratory Data Analysis(EDA) visualizations for the data are as below
![output_10_0](https://github.com/user-attachments/assets/aa66dc3e-46ba-4532-9b00-7decfcd3bc84)

##### Success Metrics

###### Below are the metrics for the successful submission of this case study.

*The accuracy of the model on the test data set should be > 75%(Subjective in nature)

*Add methods for Hyperparameter tuning.

*Perform model validation.

###### From the source code Model selection and Hyperparameter tuning metrics we can see that the success metrics of the accuracy of the model on the test data set is > 75%. So the data met the above success metrics.

###### For Model Validation the confusion matrix gives a clear understanding of how many predictions were correct and incorrect. For the data given it as shown below.

![output_26_0](https://github.com/user-attachments/assets/1a374ad0-d032-4d65-ba2e-a6d9e2720c6f)
