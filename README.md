# Employment Salary Prediction

## About

- <strong>University: </strong>University of Prishtina
- <strong>Faculty: </strong>Faculty of Electrical and Computer Engineering
- <strong>Study Program: </strong>Master of Computer and Software Engineering
- <strong>Subject (1st year): </strong>Machine Learning taught by [Prof. Dr. Eng. Lule Ahmedi](https://staff.uni-pr.edu/profile/luleahmedi) and [PhD. c Mërgim Hoti](https://staff.uni-pr.edu/profile/m%C3%ABrgimhoti)
- <strong>Students:</strong> [Festina Qorrolli](https://github.com/festinaqorrolli) and [Fisnik Spahija](https://github.com/Fisinik/)

The goal of this [project](https://github.com/fisinik/employment-salary-prediction) is to predict the male/female employment salary based on the data provided by [Tax Administration of Kosovo](https://www.atk-ks.org/en/open-data/). This project is used for the Machine Learning course in University of Prishtina, Computer and Software Engineering.

## Instructions

This project requires venv environment. This can be done by creating a workspace environment through VScode. Make sure python and pip are installed.

Install kernel for the environment by running the following command in the terminal:

```bash
pip install ipykernel
```

Install the necessary packages.

```bash
pip install -r requirements.txt
```

## Phase 1 (Preparing the model)

### Dataset overview

#### Attributes (27):

- Viti Godina Year <strong>(categorical ordinal)</strong>
- "Muaji
  Mesec
  Month" <strong>(categorical ordinal)</strong>
- "PERSHKRIMI I SEKTORIT
  OPIS SEKTORA
  SECTOR DESCRIPTION" <strong>(categorical nominal)</strong>
- "Statusi i regjistrimit
  Status registracije Registration status" <strong>(categorical nominal)</strong>
- Komuna Opstina Municipality <strong>(categorical nominal)</strong>
- "Nr Tatimp
  Poreski obveznik
  Number of Taxpayers" <strong>(numerical discrete)</strong>
- "Nr Puns
  Broj zaposlenih
  Number of employees" <strong>(numerical discrete)</strong>
- Primar Primarna Primary <strong>(numerical discrete)</strong>
- Sekondar Sekundarna Secondary <strong>(numerical discrete)</strong>
- "Meshkuj
  Muskarci
  Men (M)." <strong>(numerical discrete)</strong>
- "Femra
  Zenske
  Women (F)" <strong>(numerical discrete)</strong>
- "Pa Verif
  Neprovereno
  Unverified" <strong>(numerical discrete)</strong>
- M 15-24 <strong>(numerical discrete)</strong>
- F 15-24 <strong>(numerical discrete)</strong>
- M 25-34 <strong>(numerical discrete)</strong>
- F 25-34 <strong>(numerical discrete)</strong>
- M 35-44 <strong>(numerical discrete)</strong>
- F 35-44 <strong>(numerical discrete)</strong>
- M 45-54 <strong>(numerical discrete)</strong>
- F 45-54 <strong>(numerical discrete)</strong>
- M 55-64 <strong>(numerical discrete)</strong>
- F 55-64 <strong>(numerical discrete)</strong>
- M 65+ <strong>(numerical discrete)</strong>
- F 65+ <strong>(numerical discrete)</strong>
- "Mesat. Pages
  Prosecna plata
  Average Wage" <strong>(numerical continuous)</strong>
- "Mesat. Meshk
  Prosecni M.
  Average M." <strong>(numerical continuous)</strong>
- "Mesat. Fem.
  Prosecni F. Average F." <strong>(numerical continuous)</strong>

#### Data integration

Our dataset is separated based on years of employment data on 5 CSV (Comma Separated Values) files. The files can be found on the link provided on the about section. Using pandas python package we have read and concatinated the 5 csv files into one dataframe "employment_combined".

#### Dataset Size

The resulting dataframe consists of <strong>88014 rows</strong> spaned through 5 years divided into <strong>27 attributes</strong> as described above.

#### Null values

![Null Values](images/null-values.png)

### Dimension reduction and agreggation

Since the attributes are described in three languages, for simplicity's sake we have changed the column names to the corresponding English representation.

Noticing how columns "Men" and "Women" don't have an accurate summary of the age group columns, we have dropped the Men, Women columns and have added the accurate summaries from the age group columns.

Furthermore, columns such as Number of Taxpayers, Primary, Secondary, Unverified, Average wage, have been removed in the context of our objectives stated in the about section.

After these modifications our dataframe consists of the following columns:

- Year <strong>(categorical ordinal)</strong>
- Month <strong>(categorical ordinal)</strong>
- Month-Year <strong>(categorical ordinal)</strong>
- Sector Description <strong>(categorical nominal)</strong>
- Registration Status <strong>(categorical nominal)</strong>
- Municipality <strong>(categorical nominal)</strong>
- Number of Employees <strong>(numerical discrete)</strong>
- Men <strong>(numerical discrete)</strong>
- Women <strong>(numerical discrete)</strong>
- Average Wage Men <strong>(numerical continuous)</strong>
- Average Wage Women <strong>(numerical continuous)</strong>

![Dataframe information after dimension reduction and agreggation](images/dimension-reduction-and-aggregation.png)

### Data cleaning

By default values in columns "Average Wage Men" and "Average Wage Women" are NaN if there isn't a male/female employee in that instance. Therefore we fill these columns where values are missing with zeros only if the corresponding Men/Women column has zero value also.

After cleaning these columns our dataset consists of <strong>70611 cleaned rows</strong>.

### Discretization of "Registration Status" and "Sector Description"

As seen on the preprocessing file, we notice that column "Registration Status" has <strong>30 unique string values</strong> whereas 'Sector Description' has <strong>22 unique string values</strong>. We have created two new columns "Registration Id" and "Sector Id" to apply discretization of these columns.

![Discretization](images/discretization.png)

### Anomaly Detection

Identifying anomalies can provide valuable insights into the data, below we can see the detection of them:

![Before Anomaly Detection](images/boxplot_before_anomaly_detection.png)

After applying Isolation Forest we can see the results clearly indicate a more consistent distribution.

![After Anomaly Detection](images/boxplot_after_anomaly_detection.png)

We can see that Isolation Forest hasn't completely helped us counter the anomalies therefore we also use z-score to remove the remaining.

![After Z-Score Detection](images/boxplot_after_zscore_detection.png)

### Data Skewness

Based on the data skewness we can see that the skewness is positive.

![Data Skewness](images/data-skewness.png)

Based on skewness we can deduct that most people earn less than the average wage, with a smaller number of people earning much higher.

![Data Skewness](images/histogram_average_wage.png)

### Correlation Matrix

Representation of correlation between attributes.

![Correlation Matrix](images/correlation_matrix.png)

### SMOTE Algorithm

With SMOTE Algorithm we generate synthetic samples from the minority class (in this case where female wages are greater than male wages) to balance the dataset. This creates a more unbiased predictability performance.

![Smote Algorithm](images/smote.png)

### Splitting dataset into training and testing data

Attributes we consider on using for training model are:

- Sector Description
- Registration Status
- Municipality
- Average Wage Men
- Average Wage Women

Training set consists of <strong>42656 rows</strong>.

Testing set consists of <strong>18282 rows</strong>.

The files for testing and training can be found in /dataset:

- /dataset/testing_set.csv
- /dataset/training_set.csv

## Phase 2 (Training the model)

During training, the model learns patterns on the data related to the target variable.
It is therefore essential to analyze and compare the performance of the model against other algorithms to determine if further additional optimizations are needed.

We will analyze the training results of 7 different algorithms which are:

- Neural Network
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- K Neighbors
- Gradient Boosting Classifier

We will evaluate different data splits such as 40/60, 30/70, 20/80, 10/90 using metrics like Accuracy, F1-score, Recall, and Precision.

- Accuracy: It measures the overall correctness of a model by comparing the number of correct predictions to the total number of predictions made. Mathematically, accuracy is calculated as (TP + TN) / (TP + TN + FP + FN), where TP is True Positives, TN is True Negatives, FP is False Positives, and FN is False Negatives.

- Recall (Sensitivity or True Positive Rate): It measures the ability of a model to correctly identify positive instances from all actual positive instances. It is calculated as TP / (TP + FN), where TP is True Positives and FN is False Negatives. High recall indicates that the model is good at minimizing false negatives.

- Precision: It measures the accuracy of positive predictions made by the model. It is calculated as TP / (TP + FP), where TP is True Positives and FP is False Positives. Precision is important when the cost of false positives is high.

- F1 Score: It is the harmonic mean of precision and recall, providing a balance between the two metrics. It is calculated as 2 _ (Precision _ Recall) / (Precision + Recall). F1 score is useful when there is an uneven class distribution, as it considers both false positives and false negatives.

### Neural Network

    Accuracy: Ranges from approximately 0.714 to 0.726 across different test ratios.
    Recall: Varies from around 0.706 to 0.807, indicating the proportion of actual positives correctly identified.
    Precision: Ranges from about 0.691 to 0.730, representing the proportion of predicted positives that are correct.
    F1 Score: Varies from approximately 0.716 to 0.747, which is the harmonic mean of precision and recall.

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.721    | 0.706  | 0.730     | 0.718    |
| 30/70 | 0.726    | 0.807  | 0.696     | 0.747    |
| 20/80 | 0.720    | 0.801  | 0.691     | 0.742    |
| 10/90 | 0.714    | 0.720  | 0.712     | 0.716    |

### Logistic Regression

Performance: Consistently shows accuracy, recall, precision, and F1 scores around 0.647 across different test ratios, indicating stable but relatively lower performance compared to other models.

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.647    | 0.669  | 0.642     | 0.655    |
| 30/70 | 0.647    | 0.669  | 0.642     | 0.655    |
| 20/80 | 0.647    | 0.669  | 0.642     | 0.655    |
| 10/90 | 0.647    | 0.669  | 0.642     | 0.655    |

### Decision Tree Classifier

Accuracy, Recall, Precision, F1 Score: Both models perform consistently well across different test ratios, with accuracy around 0.882 and other metrics like recall, precision, and F1 score around 0.867 to 0.895, indicating good overall performance.

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.882    | 0.867  | 0.895     | 0.881    |
| 30/70 | 0.882    | 0.867  | 0.895     | 0.881    |
| 20/80 | 0.882    | 0.867  | 0.895     | 0.881    |
| 10/90 | 0.882    | 0.867  | 0.895     | 0.881    |

### Random Forest Classifier

Performance: SVM and K Neighbors show similar performance metrics across different test ratios, with accuracy around 0.704 and other metrics like recall, precision, and F1 score around 0.681 to 0.769, indicating moderate performance.

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.882    | 0.871  | 0.891     | 0.881    |
| 30/70 | 0.881    | 0.872  | 0.889     | 0.880    |
| 20/80 | 0.883    | 0.875  | 0.890     | 0.882    |
| 10/90 | 0.883    | 0.874  | 0.890     | 0.882    |

### Support Vector Machine

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.704    | 0.769  | 0.681     | 0.722    |
| 30/70 | 0.704    | 0.769  | 0.681     | 0.722    |
| 20/80 | 0.704    | 0.769  | 0.681     | 0.722    |
| 10/90 | 0.704    | 0.769  | 0.681     | 0.722    |

### K Neighbors

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.862    | 0.857  | 0.866     | 0.862    |
| 30/70 | 0.862    | 0.857  | 0.866     | 0.862    |
| 20/80 | 0.862    | 0.857  | 0.866     | 0.862    |
| 10/90 | 0.862    | 0.857  | 0.866     | 0.862    |

### Gradient Boosting Classifier

Performance: Shows moderate performance with accuracy around 0.746 and other metrics like recall, precision, and F1 score around 0.738 to 0.766 across different test ratios.

| Ratio | Accuracy | Recall | Precision | F1 Score |
| ----- | -------- | ------ | --------- | -------- |
| 40/60 | 0.746    | 0.766  | 0.738     | 0.752    |
| 30/70 | 0.746    | 0.766  | 0.738     | 0.752    |
| 20/80 | 0.746    | 0.766  | 0.738     | 0.752    |
| 10/90 | 0.746    | 0.766  | 0.738     | 0.752    |

## Phase 3 (Analysis and Evaluation)

All models were trained and evaluated using a consistent methodology, including data preprocessing steps such as one-hot encoding and standardization. The models were evaluated on various metrics to provide a comprehensive view of their performance.

### Performance Metrics

- Logistic Regression and Decision Tree Classifier showed excellent performance with high accuracy, recall, precision, and F1 scores across different train-test splits.
- Random Forest Classifier also performed well, especially in terms of recall and F1 score, indicating its robustness.
- Support Vector Machine and K-Neighbors showed relatively lower performance compared to the other models, although they still achieved reasonable accuracy and recall.
- Gradient Boosting Classifier performed consistently across different splits, indicating its effectiveness in handling imbalanced data.

### Neural Network

A neural network is composed of layers of neurons, where each neuron applies a linear transformation followed by a non-linear activation function. The network learns by adjusting the weights through backpropagation to minimize the loss function. This model's strength is due to its' capability of capturing complex non-linear relationships and patterns in the data.

- Accuracy increased significantly, reaching around 0.9713 to 0.9773.
- Recall improved, now consistently high, around 0.9713 to 0.9773.
- Precision saw substantial improvement, ranging from 0.9723 to 0.9775.
- F1 Score also improved notably, with values now between 0.9716 to 0.9774.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.714 - 0.726   | 0.706 - 0.807   | 0.691 - 0.730   | 0.716 - 0.747   |
| Optimized   | 0.9713 - 0.9773 | 0.9713 - 0.9773 | 0.9723 - 0.9775 | 0.9716 - 0.9774 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9712832570075989 | 0.9712832294059731 | 0.9722647490420612 | 0.9715646480520717 |
| 30/70 | 0.9773000478744507 | 0.977300076578055  | 0.9774782716295978 | 0.9773663995005927 |
| 20/80 | 0.97489333152771   | 0.9748933377092222 | 0.9756684549996559 | 0.9751025475532312 |
| 10/90 | 0.9732523560523987 | 0.9732523793895635 | 0.9737151190851275 | 0.973239009894511  |

### Logistic Regression

A linear model that estimates the probability of a binary outcome using the logistic function. It calculates the log-odds of the outcome as a linear combination of the input features. It's simple, interpretable, and effective for binary classification tasks.

- Accuracy improved dramatically to around 0.9979 - 0.9983.
- Recall increased significantly, now in the range of 0.9801 to 0.9877.
- Precision improved, ranging from 0.9847 to 0.9879.
- F1 Score saw substantial improvement, now between 0.9834 to 0.9864.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.647           | 0.669           | 0.642           | 0.655           |
| Optimized   | 0.9979 - 0.9983 | 0.9801 - 0.9877 | 0.9847 - 0.9879 | 0.9834 - 0.9864 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9981949458483754 | 0.9841479524438573 | 0.9867549668874173 | 0.9854497354497355 |
| 30/70 | 0.9983043430696861 | 0.9876651982378855 | 0.9850615114235501 | 0.9863616366036076 |
| 20/80 | 0.9980308500164096 | 0.980106100795756  | 0.9879679144385026 | 0.9840213049267643 |
| 10/90 | 0.9978667541844437 | 0.9821428571428571 | 0.9846547314578005 | 0.983397190293742  |

### Decision Tree Classifier

A tree structure where each node represents a feature, and branches represent decisions based on the feature's value. The tree splits the data recursively to minimize impurity. Easy to interpret and understand, handles both numerical and categorical data well.

- Accuracy improved significantly, reaching around 0.9975 to 0.9980.
- Recall saw a notable increase, now between 0.9775 to 0.9949.
- Precision improved to around 0.9750 to 0.9854.
- F1 Score increased, now ranging from 0.9801 to 0.9848.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.882           | 0.867           | 0.895           | 0.881           |
| Optimized   | 0.9975 - 0.9980 | 0.9775 - 0.9949 | 0.9750 - 0.9854 | 0.9801 - 0.9848 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9980308500164096 | 0.9828269484808454 | 0.9854304635761589 | 0.984126984126984  |
| 30/70 | 0.9977573569631332 | 0.9814977973568282 | 0.982363315696649  | 0.9819303657999118 |
| 20/80 | 0.997538562520512  | 0.9774535809018567 | 0.9826666666666667 | 0.9800531914893618 |
| 10/90 | 0.9980308500164096 | 0.9948979591836735 | 0.975              | 0.984848484848485  |

### Random Forest Classifier

An ensemble of decision trees where each tree is trained on a random subset of the data. The final prediction is made by averaging the predictions of all trees. Reduces overfitting, robust to noise, and provides high accuracy.

- Accuracy saw significant improvement, now around 0.9943 to 0.9947.
- Recall increased notably, ranging from 0.9346 to 0.9541.
- Precision improved, now between 0.9624 to 0.9725.
- F1 Score saw substantial improvement, reaching around 0.9532 to 0.9590.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.881 - 0.883   | 0.871 - 0.875   | 0.889 - 0.891   | 0.880 - 0.882   |
| Optimized   | 0.9943 - 0.9947 | 0.9346 - 0.9541 | 0.9624 - 0.9725 | 0.9532 - 0.9590 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9942976698391861 | 0.9346103038309115 | 0.9725085910652921 | 0.9531828898619065 |
| 30/70 | 0.9945848375451264 | 0.9418502202643172 | 0.97005444646098   | 0.9557443004023245 |
| 20/80 | 0.9945848375451264 | 0.9496021220159151 | 0.9623655913978495 | 0.9559412550066756 |
| 10/90 | 0.9947489333770922 | 0.9540816326530612 | 0.9639175257731959 | 0.958974358974359  |

### Support Vector Machine

Finds the optimal hyperplane that maximizes the margin between different classes. SVM can be extended to non-linear classification using kernel functions. Effective in high-dimensional spaces, robust to overfitting in low-dimensional space.

- Accuracy improved dramatically, now around 0.9891 to 0.9926.
- Recall increased significantly, ranging from 0.8851 to 0.9388.
- Precision saw a substantial improvement, now between 0.9358 to 0.9460.
- F1 Score improved notably, reaching around 0.9097 to 0.9424.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.704           | 0.769           | 0.681           | 0.722           |
| Optimized   | 0.9891 - 0.9926 | 0.8851 - 0.9388 | 0.9358 - 0.9460 | 0.9097 - 0.9424 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9890876271742698 | 0.8850726552179656 | 0.9357541899441341 | 0.9097080787508486 |
| 30/70 | 0.9897166611968056 | 0.8951541850220265 | 0.936405529953917  | 0.9153153153153153 |
| 20/80 | 0.991384968821792  | 0.9151193633952255 | 0.9439124487004104 | 0.9292929292929293 |
| 10/90 | 0.992615687561536  | 0.9387755102040817 | 0.9460154241645244 | 0.942381562099872  |

### K Neighbors

Classifies a data point based on the majority class among its k-nearest neighbors in the feature space. Simple and intuitive, effective with well-separated classes.

- Accuracy improved significantly, now around 0.9833 to 0.9872.
- Recall increased, ranging from 0.8157 to 0.8852.
- Precision saw an improvement, now between 0.9054 to 0.9132.
- F1 Score improved notably, reaching around 0.8582 to 0.8990.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.862           | 0.857           | 0.866           | 0.862           |
| Optimized   | 0.9833 - 0.9872 | 0.8157 - 0.8852 | 0.9054 - 0.9132 | 0.8582 - 0.8990 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9832622251394815 | 0.8157199471598415 | 0.905425219941349  | 0.8582348853370396 |
| 30/70 | 0.9849578820697954 | 0.8387665198237886 | 0.9118773946360154 | 0.8737953189536485 |
| 20/80 | 0.9868723334427305 | 0.8673740053050398 | 0.9159663865546218 | 0.891008174386921  |
| 10/90 | 0.9872005251066623 | 0.8852040816326531 | 0.9131578947368421 | 0.8989637305699482 |

### Gradient Boosting Classifier

An ensemble method that builds models sequentially, where each new model corrects the errors of the previous ones. It uses a gradient descent algorithm to minimize the loss. Powerful for both regression and classification tasks, capable of capturing complex patterns and relationships.

- Accuracy improved dramatically, now around 0.9972 to 0.9980.
- Recall increased significantly, ranging from 0.9795 to 0.9828.
- Precision saw substantial improvement, now between 0.9771 to 0.9880.
- F1 Score improved notably, reaching around 0.9783 to 0.9841.

| Performance | Accuracy        | Recall          | Precision       | F1 Score        |
| ----------- | --------------- | --------------- | --------------- | --------------- |
| Initial     | 0.746           | 0.766           | 0.738           | 0.752           |
| Optimized   | 0.9972 - 0.9980 | 0.9795 - 0.9828 | 0.9771 - 0.9880 | 0.9783 - 0.9841 |

| Ratio | Accuracy           | Recall             | Precision          | F1 Score           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ |
| 40/60 | 0.9979898260584181 | 0.9795244385733157 | 0.9880079946702198 | 0.9837479270315092 |
| 30/70 | 0.9974291652992014 | 0.9779735682819384 | 0.980565371024735  | 0.9792677547419498 |
| 20/80 | 0.9980308500164096 | 0.9827586206896551 | 0.9853723404255319 | 0.9840637450199203 |
| 10/90 | 0.9972103708565803 | 0.9795918367346939 | 0.9770992366412213 | 0.9783439490445859 |

### Conclusion

The optimization process resulted in substantial improvements across all machine learning models, highlighting the importance of fine-tuning and parameter adjustments. Logistic Regression and Gradient Boosting Classifier, in particular, achieved exemplary performance metrics, making them the preferred choices for this prediction task. Decision Tree and Random Forest Classifiers also demonstrated reliable and robust performance, suitable for scenarios requiring high accuracy and interpretability. Support Vector Machine and K Neighbors, although improved, might be considered for specific contexts where their unique advantages apply.
