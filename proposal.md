# Machine Learning Engineer Nanodegree
## Capstone Proposal
Mincent Lee  
31 January  2018

## Proposal

### Domain Background

This project will develop a stock price predictor by machine learning. The proposal is historically simplified from the [Project Description ~ Investment and Trading](http://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)[^descript] and based on the [Course ~ Machine Learning for Trading](http://udacity.com/course/machine-learning-for-trading--ud501)[^course] for my first solid step to study machine learning for trading. Because the risk free rate of return from a bank account or a very short-term treasury bond is about 0 lately, folks have put so much money into the stock market[^course]. The stock prediction can help us to understand market behaviour and trade profitable investments according to the wealthy information in the stock history and company data which is suitable for machine learning process[^descript]. There are lot related academic research support the stock prediction[^course][^predict] while there are also opponent Efficient-Market Hypothesis[^hypo].

[^descript]: ["MLND Capstone Project Description - Investment and Trading," *Udacity*](http://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)

[^course]: [Tucker Balch, "Machine Learning for Trading," *Georgia Tech* and *Udacity*](http://udacity.com/course/machine-learning-for-trading--ud501)

[^predict]: ["Stock market prediction," *Wikipedia*](http://wikipedia.org/wiki/Stock_market_prediction)

[^hypo]: ["Efficient-market hypothesis," *Wikipedia*](http://en.wikipedia.org/wiki/Efficient-market_hypothesis)

The datasets used in this project is obtained from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance][^fix]. The target stock might be the [S&P 500 Index](http://wikipedia.org/wiki/S%26P_500_Index)[^sp] that might be the best representation of the U.S. stock market[^sp]. The inputs include *Opening price*, *Highest price*, *traded Volume*, *Adjusted Closing price*, and so on. 
Each price prediction is according to the trading data of a consistent **day range**, e.g., considering 2+1-day range in a trading week, the input ($X_1, X_2, X_3...$) and predicted ($y_1, y_2, y_3...$) days are:

| |X (2-day range)|y (the next day of the range)|
|-|---------------|-----------------------------|
|1|Mon. Tue.      |Wed.
|2|Tue. Wed.      |Thu.
|3|Wed. Thu.      |Fri.

The sampled days for this project should include the current day for practicality and then trace back to find a balanced day range in which the distribution of the target classes (price *Rise*/*Fall*) is balanced for balanced evaluation metrics. The balanced day range could be searched from the same-price ranges in which the prices of the first and last day are the same to have balanced probability of *Rises* and *Falls*. The sampled day range might also larger than one year to cover annual and monthly characteristics. The first experiment is planned to train with the data last year (Jan. 2017 to Dec. 2017) and test this year (Jan. 2018 to Mar. 2018).

[^yahoo]: [Yahoo! Finance](http://finance.yahoo.com)

[^finance]: ["yahoo-finance 1.4.0," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/yahoo-finance)

[^fix]: ["fix-yahoo-finance 0.0.21," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/fix-yahoo-finance)

[^sp]: ["Standard & Poor's 500," *Wikipedia*](http://wikipedia.org/wiki/S%26P_500_Index)

### Problem Statement

For reality and accuracy[^hypo] concerns, the target problem of my first stock study is simplified to predict whether the *Adjusted* (for stock splits and dividends) *Closing price* rises or falls. The stock price predictor is inputted a certain range of daily trading data and outputs whether the *Adjusted Closing price* rises or falls (might ignore the rare flat cases at the first step) next to the certain range, i.e., the predicted day is the *next day*, e.g., predicting the last Thursday according to the data of the last Monday to Wednesday. The next day is supposed to have the highest correlation and predictability according to the input features, and suitable to be the basic first step. This is quantifiable, measurable, and replicable. The relevant potential solution are the Classifiers of the [scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [ensemble Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC].

[^map]: ["Choosing the right estimator," *scikit-learn.org*](http://scikit-learn.org/stable/tutorial/machine_learning_map)

[^GBC]: ["sklearn.ensemble.GradientBoostingClassifier," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

### Datasets and Inputs

### Solution Statement

The potential solution is training a Classifier by daily trading data within specific ranges of days to predict *Rising* or *Falling* of the *Adjusted Closing price* following the range. The daily trading data are obtained from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance]. The machine learning libraries and Classifiers might come from [scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [ensemble Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC], and the parameter [random_state](http://scikit-learn.org/stable/developers/utilities.html)[^rand] will be recorded. Therefore, the solution is quantifiable, measurable, and replicable.

[^rand]: ["Utilities for Developers," *scikit-learn.org*](http://scikit-learn.org/stable/developers/utilities.html)

### Benchmark Model

The benchmark model will be the [k-nearest neighbors vote](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)[^k] or [C-Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[^svc] depended on further experimental results.
~~The predicted *Rising* and *Falling* results will be evaluated in fact with the exact benchmark of specific daily prices from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance]. Might also build a [Naïve Predictor](http://github.com/udacity/machine-learning/blob/master/projects/finding_donors/finding_donors.ipynb)[^naïve] which always predict *True*/*False* (*Rising*/*Falling*), if necessary.~~

[^k]: ["sklearn.neighbors.KNeighborsClassifier," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

[^svc]: ["sklearn.svm.SVC," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

[^naïve]: [udacity, "Project: Finding Donors for CharityML," *github.com*](http://github.com/udacity/machine-learning/blob/master/projects/finding_donors/finding_donors.ipynb)

### Evaluation Metrics

The solution model will be evaluated with the exact benchmark of specific daily prices from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the [$F_{\beta}-score$](http://wikipedia.org/wiki/F1_score)[^f1] with the [fbeta_score function of scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)[^beta]. The mathematical representations is:
$$F_{\beta} = (1+\beta^{2})\tfrac{precision\cdot recall}{\beta^2precision+recall}$$
The $\beta$ might be 1 for balanced precision and recall[^f1].

[^f1]: ["F1 score," *Wikipedia*](http://wikipedia.org/wiki/F1_score)

[^beta]: ["sklearn.metrics.fbeta_score," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

### Project Design

- Analysis
  - Data Collection
    - Will try [S&P 500 Index](http://wikipedia.org/wiki/S%26P_500_Index)[^sp] first
    - Data include *Opening price*, *Highest price*, *traded Volume*, *Adjusted Closing price*, and so on
    - Will try the python modules
      - ["yahoo-finance 1.4.0," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/yahoo-finance)[^finance]
      - ["fix-yahoo-finance 0.0.21," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/fix-yahoo-finance)[^fix]
  - Data Structure
    - Format data to the [DataFrame of pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)[^pd]
  - Data Cleaning
    - Basic abnormal trading data handling[^course]
    - [Imputation of missing values](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)[^miss]
  - Feature-set Exploration
    - Feature-set include *Opening price*, *Highest price*, *traded Volume*, *Adjusted Closing price*, and so on
    - Data Mining
      - [pandas.DataFrame.describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)[^describe]
      - [NumPy](http://numpy.org)[^np]
    - Exploratory Visualization
      - [matplotlib](http://matplotlib.org)[^mat]
      - [seaborn ~ statistical data visualization](http://seaborn.pydata.org)[^sb]
  - Benchmark
    - Models
      - The benchmark model will be the [k-nearest neighbors vote](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)[^k] or [C-Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[^svc] depended on further experimental results.
      - ~~Exact benchmark of specific daily prices from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance]~~
      - ~~Might build a [Naïve Predictor](http://github.com/udacity/machine-learning/blob/master/projects/finding_donors/finding_donors.ipynb)[^naïve] which always predict *True*/*False* (*Rising*/*Falling*), if necessary~~
    - Evaluation Metrics
      - [$F_{\beta}-score$](http://wikipedia.org/wiki/F1_score)[^f1]
        - [fbeta_score function of scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)[^beta]
        - The $\beta$ might be 1 for balanced precision and recall[^f1]
- Methodology
  - Data Pre-processing
    - Outlier detection
      - [Methods](http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html)/[algorithms of scikit-learn](http://scikit-learn.org/dev/auto_examples/plot_anomaly_comparison.html)[^outlier][^anomaly]
    - Normalizing Numerical Features
      - [Scaling/Standardization](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)[^scaling] [scalers of scikit-learn](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)[^scaler]
    - Encode Stock Price Changings for Classification
      - Encode the predicted *Adjusted Closing price* to *True* if it is *Rising* than the previous trading day (skip the non-trading days) and vice versa (*False* if it is *Falling*)
    - ~~Shuffle and Split Data~~
      - ~~Apply [sklearn.model_selection.train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)[^split] with recorded random_state[^rand] for replicability~~
      - The *Shuffle* might be avoided to avoid the [look ahead bias in time series](http://datasciencecentral.com/profiles/blogs/avoiding-look-ahead-bias-in-time-series-modelling-1)[^bias] as the reviewed comments.
      - The training date range might be the last year and the testing day range might be this year.
      - However, there is a *Shuffle* experiment that the data are constructed into many isolated day range packages in which the days inside are kept continuous for each prediction, e.g., considering 2+1-day range packages in a trading week, the input ($X_1, X_2, X_3...$) and predicted ($y_1, y_2, y_3...$) days are as the table below. Therefore, the prediction packages can keep continuous day range inside, and the outside index 1~3 can be shuffled.

| |X (2-day range)|y (the next day of the range)|
|-|---------------|-----------------------------|
|1|Mon. Tue.      |Wed.
|2|Tue. Wed.      |Thu.
|3|Wed. Thu.      |Fri.

  - Supervised Learning Models
    - [Classifiers in scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [ensemble Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC]
  - Training and Predicting Pipeline
    - Build normalizing, training, predicting, scoring functions
    - Make the normalizing [Scaler](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)[^scaler] and [Classifier](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map] in a [Pipeline](http://scikit-learn.org/stable/modules/pipeline.html)[^pipe]
  - Initial Model Evaluation
    - Apply default and coarse-grained parameters
    - Record all the available parameter random_state[^rand] for replicability
    - The specified day range of each daily trading data might be a week initially
  - Refinement
    - [Fine-tune the hyper-parameters](http://scikit-learn.org/stable/modules/grid_search.html)[^hyper] ([Exhaustive Grid Search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)[^hyper] by [Cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html)[^cv] with [TimeSeriesSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)[^time])
    - Tune the day range of each input might by [Feature importances](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)[^import] or [Feature selection](http://scikit-learn.org/stable/modules/feature_selection.html)[^select]
- Results
  - Model Evaluation and Validation
    - Evaluate with the [$F_{1}-score$](http://wikipedia.org/wiki/F1_score)[^f1]
    - Validate by the [Cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html)[^cv] with the [TimeSeriesSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)[^time] and long period of days
  - Justification 
    - Compare with the exact benchmark of daily prices[^finance] in fact
    - Might compare with the [Naïve Predictor](http://github.com/udacity/machine-learning/blob/master/projects/finding_donors/finding_donors.ipynb)[^naïve], if necessary

[^pd]: ["pandas.DataFrame," *Python Data Analysis Library, pandas.pydata.org*](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)

[^describe]: ["pandas.DataFrame.describe," *pandas.pydata.org*](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)

[^miss]: ["Imputation of missing values," *scikit-learn.org*](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)

[^np]: [NumPy, *SciPy.org*](http://www.numpy.org)

[^mat]: [matplotlib.org](http://matplotlib.org)

[^sb]: [seaborn: statistical data visualization, *seaborn.pydata.org*](http://seaborn.pydata.org)

[^anomaly]: ["Comparing anomaly detection algorithms for outlier detection on toy datasets," *scikit-learn.org*](http://scikit-learn.org/dev/auto_examples/plot_anomaly_comparison.html)

[^outlier]: ["Outlier detection with several methods," *scikit-learn.org*](http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html)

[^scaling]: ["Standardization, or mean removal and variance scaling," *scikit-learn.org*](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

[^scaler]: ["Compare the effect of different scalers on data with outliers," *scikit-learn.org*](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)

[^split]: ["sklearn.model_selection.train_test_split," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

[^pipe]: ["Pipeline: chaining estimators," *scikit-learn.org*](http://scikit-learn.org/stable/modules/pipeline.html)

[^hyper]: ["Tuning the hyper-parameters of an estimator," *scikit-learn.org*](http://scikit-learn.org/stable/modules/grid_search.html)

[^cv]: ["Cross-validation: evaluating estimator performance," *scikit-learn.org*](http://scikit-learn.org/stable/modules/cross_validation.html)

[^import]: ["Feature importances with forests of trees," *scikit-learn.org*](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

[^select]: ["Feature selection," *scikit-learn.org*](http://scikit-learn.org/stable/modules/feature_selection.html)

[^bias]: [Rohit Walimbe, "Avoiding Look Ahead Bias in Time Series Modelling," *datasciencecentral.com*](http://datasciencecentral.com/profiles/blogs/avoiding-look-ahead-bias-in-time-series-modelling-1)

[^time]: ["sklearn.model_selection.TimeSeriesSplit," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)