# Machine Learning Engineer Nanodegree
## Capstone Proposal
Mincent Lee
January 31st, 2018

## Proposal

### Domain Background

This project will develop a stock price predictor by machine learning. The proposal is historically simplified from the [Project Description ~ Investment and Trading](http://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)[^descript] and based on the [Course ~ Machine Learning for Trading](http://udacity.com/course/machine-learning-for-trading--ud501)[^course] for my first solid step to study machine learning for trading. Because the risk free rate of return from a bank account or a very short-term treasury bond is about 0 lately, folks have put so much money into the stock market[^course]. The stock prediction can help us to understand market behaviour and trade profitable investments according to the wealthy information in the stock history and company data which is suitable for machine learning process[^descript]. There are lot related academic research support the stock prediction[^course][^predict] while there are also opponent Efficient-Market Hypothesis[^hypo].

[^descript]: ["MLND Capstone Project Description - Investment and Trading," *Udacity*](http://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)

[^course]: [Tucker Balch, "Machine Learning for Trading," *Georgia Tech* and *Udacity*](http://udacity.com/course/machine-learning-for-trading--ud501)

[^predict]: ["Stock_market_prediction," *Wikipedia*](http://wikipedia.org/wiki/Stock_market_prediction)

[^hypo]: ["Efficient-market hypothesis," *Wikipedia*](http://en.wikipedia.org/wiki/Efficient-market_hypothesis)

### Problem Statement

For reality and accuracy[^hypo] concerns, the target problem of my first stock study is simplified to predict whether the adjusted (for stock splits and dividends) closing price rises or falls. The stock price predictor is inputted a certain range of daily trading data and outputs whether the adjusted closing price rises or falls (might ignore the rare flat cases at the first step) next to the certain range. This is quantifiable, measurable, and replicable. The relevant potential solution are the classifiers of the [scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [ensemble gradient boosting classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC].

[^map]: ["Choosing the right estimator," *scikit-learn.org*](http://scikit-learn.org/stable/tutorial/machine_learning_map)

[^GBC]: ["sklearn.ensemble.GradientBoostingClassifier," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

### Datasets and Inputs

The datasets used in this project is obtained from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance][^fix]. The target stock might be the [S&P 500 Index](http://wikipedia.org/wiki/S%26P_500_Index)[^sp] that might be the best representation of the U.S. stock market[^sp]. The inputs include opening price, highest price, traded volume, adjusted closing price, and so on.

[^yahoo]: [Yahoo! Finance](http://finance.yahoo.com)

[^finance]: ["yahoo-finance 1.4.0," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/yahoo-finance)

[^fix]: ["fix-yahoo-finance 0.0.21," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/fix-yahoo-finance)

[^sp]: ["Standard & Poor's 500," *Wikipedia*](http://wikipedia.org/wiki/S%26P_500_Index)

### Solution Statement

The potential solution is training a classifier by daily trading data within specific ranges of days to predict rising or falling of the adjusted closing price following the range. The daily trading data are obtained from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance]. The machine learning libraries and classifiers might come from [scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [ensemble gradient boosting classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC], and the parameter [random_state](http://scikit-learn.org/stable/developers/utilities.html)[^rand] will be recorded. Therefore, the solution is quantifiable, measurable, and replicable.

[^rand]: ["Utilities for Developers," *scikit-learn.org*](http://scikit-learn.org/stable/developers/utilities.html)

### Benchmark Model

The predicted rising and falling results will be evaluated in fact with the exact benchmark of specific daily prices from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the python module [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^finance].

### Evaluation Metrics

The solution model will be evaluated with the exact benchmark of specific daily prices from the [Yahoo! Finance](http://finance.yahoo.com)[^yahoo] by the [F-score](http://wikipedia.org/wiki/F1_score)[^f1] with the [fbeta_score function of scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)[^beta]. The mathematical representations of the [F-score](http://wikipedia.org/wiki/F1_score)[^f1] is:
$$F_{\beta} = (1+\beta^{2})\tfrac{precision\cdot recall}{\beta^2precision+recall}$$
The $\beta$ might be 1 for balanced precision and recall[^f1].

[^f1]: ["F1 score," *Wikipedia*](http://wikipedia.org/wiki/F1_score)

[^beta]: ["sklearn.metrics.fbeta_score," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

### Project Design

- Data Collection
  - Will try [S&P 500 Index](http://wikipedia.org/wiki/S%26P_500_Index)[^sp] first
  - Data include opening price, highest price, traded volume, adjusted closing price, and so on
  - Will try the python modules
    - ["yahoo-finance 1.4.0," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/yahoo-finance)[^finance]
    - ["fix-yahoo-finance 0.0.21," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/fix-yahoo-finance)[^fix]
- Data Structure
  - Format data to the [DataFrame of pandas](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)[^pd]
- Data Cleaning
  - Basic abnormal trading data handling[^course]
  - [Imputation of missing values](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)[^miss]
- Feature-set Exploration
  - Feature-set include opening price, highest price, traded volume, adjusted closing price, and so on
  - Data Mining
    - [pandas.DataFrame.describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)[^describe]
    - [NumPy](http://numpy.org)[^np]
  - Exploratory Visualization
    - [matplotlib](http://matplotlib.org)[^mat]
    - [seaborn ~ statistical data visualization](http://seaborn.pydata.org)[^sb]
- Data Pre-processing
  - Outlier detection
    - [Methods](http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html)/[algorithms of scikit-learn](http://scikit-learn.org/dev/auto_examples/plot_anomaly_comparison.html)[^outlier][^anomaly]
  - Normalizing Numerical Features
    - [Scaling/Standardization](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)[^scaling] [scalers of scikit-learn](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)[^scaler]
  - Encode Stock Price Changings for Classification
    - Encode the 
  - [scikit-learn](http://scikit-learn.org)[^map]
  - Algorithms and Techniques
  - Benchmark
- Methodology
  - Implementation
  - Refinement
- Results
  - Model Evaluation and Validation
  - Justification
- Conclusion
  - Free-Form Visualization
  - Reflection
  - Improvement

[^pd]: [DataFrame, *Python Data Analysis Library, pandas.pydata.org*](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)

[^describe]: [pandas.DataFrame.describe, *pandas.pydata.org*](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)

[^miss]: ["Imputation of missing values," *scikit-learn.org*](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)

[^np]: [NumPy, *SciPy.org*](http://www.numpy.org)

[^mat]: [matplotlib.org](http://matplotlib.org)

[^sb]: [seaborn: statistical data visualization, *seaborn.pydata.org*](http://seaborn.pydata.org)

[^anomaly]: ["Comparing anomaly detection algorithms for outlier detection on toy datasets," *scikit-learn.org*](http://scikit-learn.org/dev/auto_examples/plot_anomaly_comparison.html)

[^outlier]: ["Outlier detection with several methods," *scikit-learn.org*](http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html)

[^scaling]: ["standardization-or-mean-removal-and-variance-scaling," *scikit-learn.org*](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

[^scaler]: ["Compare the effect of different scalers on data with outliers," *scikit-learn.org*](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)
