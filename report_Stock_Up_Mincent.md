# Machine Learning Capstone Project
Mincent Lee, 13 April  2018

## I. Definition
### Project Overview

#### _Domain Background_
This project will develop a stock price predictor by machine learning. The proposal is historically simplified from the [Project Description ~ Investment and Trading](http://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)[^descript] and based on the [Course ~ Machine Learning for Trading](http://udacity.com/course/machine-learning-for-trading--ud501)[^course] for my first solid step to study machine learning for trading. Because the risk free rate of return from a bank account or a very short-term treasury bond is about 0 lately, folks have put so much money into the stock market[^course]. The stock prediction can help us to understand market behaviour and trade profitable investments according to the wealthy information in the stock history and company data which is suitable for machine learning process[^descript]. There are lot related academic research support the stock prediction[^course][^predict] while there are also opponent Efficient-Market Hypothesis[^hypo].

[^descript]: ["MLND Capstone Project Description - Investment and Trading," *Udacity*](http://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)

[^course]: [Tucker Balch, "Machine Learning for Trading," *Georgia Tech* and *Udacity*](http://udacity.com/course/machine-learning-for-trading--ud501)

[^predict]: ["Stock market prediction," *Wikipedia*](http://wikipedia.org/wiki/Stock_market_prediction)

[^hypo]: ["Efficient-market hypothesis," *Wikipedia*](http://en.wikipedia.org/wiki/Efficient-market_hypothesis)

#### _Datasets and Inputs_
The datasets used in this project is obtained by the python module [googlefinance.client](http://pypi.python.org/pypi/googlefinance.client)[^goog] instead of the planned popular [yahoo-finance](http://pypi.python.org/pypi/yahoo-finance)[^yhoo] [which is being discontinued](http://yahoo.sdx.socialdynamx.com/portal/conversation/19248672)[^dis].

[^goog]: ["googlefinance.client," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/googlefinance.client)

[^yhoo]: ["yahoo-finance 1.4.0," *PyPI - the Python Package Index*](http://pypi.python.org/pypi/yahoo-finance)

[^dis]: [YahooCare, "the service is being discontinued," *yahoo.sdx.socialdynamx.com*](http://yahoo.sdx.socialdynamx.com/portal/conversation/19248672)

The target stock might be the [S&P 500 Index](http://wikipedia.org/wiki/S%26P_500_Index)[^sp] that might be the best representation of the U.S. stock market[^sp]. The inputs include daily *Opening price*, *Highest price*, *traded Volume*, *Closing price*, and so on. 
Each price prediction is according to the trading data of a consistent **day range**, e.g., considering 2+1-day range in a trading week, the input ($X_1, X_2, X_3...$) and predicted ($y_1, y_2, y_3...$) days are:
<font size="1" style="line-height:11px;letter-spacing:0px">
| |X (2-day range)|y (the next day of the range)|
|-|---------------|-----------------------------|
|1|Mon. Tue.      |Wed.                         |
|2|Tue. Wed.      |Thu.                         |
|3|Wed. Thu.      |Fri.                         |
</font>
The sampled days for this project should include the current day for practicality and then trace back to find a balanced day range in which the distribution of the target classes (price *Up*/*Down*) is balanced for balanced evaluation metrics. The balanced day range could be searched from the same-price ranges in which the prices of the first and last day are the same to have balanced probability of *Ups* and *Downs*. The sampled day range might also larger than one year to cover annual and monthly characteristics. The first experiment is planned to train with the data last year (Jan. 2017 to Dec. 2017) and test this year (Jan. 2018 to Apr. 2018).

[^sp]: ["Standard & Poor's 500," *Wikipedia*](http://wikipedia.org/wiki/S%26P_500_Index)

### Problem Statement
#### _Problem Define_
For reality and accuracy[^hypo] concerns, the target problem of my first stock study is simplified to predict whether the *Closing price* ups or downs. The stock price predictor is inputted a certain range of daily trading data and outputs whether the *Closing price* ups or downs (might ignore the rare flat cases at the first step) next to the certain range, i.e., the predicted day is the *next day*, e.g., predicting the last Thursday according to the data of the last Monday to Wednesday. The next day is supposed to have the highest correlation and predictability according to the input features, and suitable to be the basic first step. This is quantifiable, measurable, and replicable. The relevant potential solution are the Classifiers of the [scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [Ensemble Tree Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC].

[^map]: ["Choosing the right estimator," *scikit-learn.org*](http://scikit-learn.org/stable/tutorial/machine_learning_map)

[^GBC]: ["sklearn.ensemble.GradientBoostingClassifier," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

#### _Strategy_
The potential solution is training a Classifier by daily trading data within specific ranges of days to predict *Upping* or *Downing* of the *Closing price* following the range. The daily trading data are obtained from the python module [googlefinance.client](http://pypi.python.org/pypi/googlefinance.client)[^goog]. The machine learning libraries and Classifiers might come from [scikit-learn](http://scikit-learn.org/stable/tutorial/machine_learning_map)[^map], e.g., the  [Ensemble Tree Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC], and the parameter [random_state](http://scikit-learn.org/stable/developers/utilities.html)[^rand] will be recorded. Therefore, the solution is quantifiable, measurable, and replicable.

[^rand]: ["Utilities for Developers," *scikit-learn.org*](http://scikit-learn.org/stable/developers/utilities.html)

### Metrics
The solution model will be evaluated with the exact benchmark of specific daily prices from the python module [googlefinance.client](http://pypi.python.org/pypi/googlefinance.client)[^goog] by the [$F_{\beta}-score$](http://wikipedia.org/wiki/F1_score)[^f1] with the [fbeta_score function of scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)[^beta]. The mathematical representations is:<font size="1" style="line-height:11px;letter-spacing:0px">
$$F_{\beta} = (1+\beta^{2})\tfrac{precision\cdot recall}{\beta^2precision+recall}$$
</font>
  
The $\beta$ might be 1 for balanced precision and recall[^f1].

[^f1]: ["F1 score," *Wikipedia*](http://wikipedia.org/wiki/F1_score)

[^beta]: ["sklearn.metrics.fbeta_score," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

## II. Analysis

### Data Exploration
#### _Raw Data_
Import 7-year data of the [S&P 500 Index](http://wikipedia.org/wiki/S%26P_500_Index), till the showed current day below.
The columns of this dataset will be calculated to our target labels (the next day price ups, flats or downs) for each day.
The column or index names from the two functions are needs to be cleaned respectively.
<font size="1" style="line-height:11px;letter-spacing:0px" face="arial narrow">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-14 04:00:00</th>
      <td>2676.9</td>
      <td>2680.26</td>
      <td>2645.05</td>
      <td>2656.3</td>
      <td>1823268887</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>.INX_Open</th>
      <th>.INX_High</th>
      <th>.INX_Low</th>
      <th>.INX_Close</th>
      <th>.INX_Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-14</th>
      <td>2676.9</td>
      <td>2680.26</td>
      <td>2645.05</td>
      <td>2656.3</td>
      <td>1823268887</td>
    </tr>
  </tbody>
</table>
</font>

#### _Data Cleaning_
The column names are cleaned and the data with abnormal 0 also need to be checked
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1099.230000</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</font>
  
Check the 0 `Volume`.
The early years lack `Volume` data need to be cleaned.
![png](fig/0Volumn.png)

The cleaned data with complete `Volume` start from 2012-01-15.
However, there are still 0 prices need to be checked.
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
      <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1278.040000</td>
      <td>1.839316e+08</td>
    </tr>
    </tbody>
</table>
</font>

Check the data distributions.
The normal prices are over 1000.
![png](fig/0Price.png)

There is only one abnormal day needs to be dropped
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-08-01</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2470.3</td>
      <td>2189633778</td>
    </tr>
  </tbody>
</table>
</font>

The data statistics and distributions are clean
The `Volume` values need Log-transform.
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1569.000000</td>
      <td>1569.000000</td>
      <td>1569.000000</td>
      <td>1569.000000</td>
      <td>1.569000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1965.500554</td>
      <td>1974.049184</td>
      <td>1956.447081</td>
      <td>1966.028113</td>
      <td>9.016781e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>378.856947</td>
      <td>379.540049</td>
      <td>377.911261</td>
      <td>378.553520</td>
      <td>6.692904e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1277.820000</td>
      <td>1282.550000</td>
      <td>1266.740000</td>
      <td>1278.040000</td>
      <td>1.839316e+08</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1675.260000</td>
      <td>1683.730000</td>
      <td>1663.520000</td>
      <td>1676.120000</td>
      <td>4.982664e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2003.660000</td>
      <td>2018.190000</td>
      <td>1992.540000</td>
      <td>2002.330000</td>
      <td>5.795952e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2165.640000</td>
      <td>2171.360000</td>
      <td>2157.090000</td>
      <td>2164.450000</td>
      <td>8.658505e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2867.230000</td>
      <td>2872.870000</td>
      <td>2851.480000</td>
      <td>2872.870000</td>
      <td>4.024144e+09</td>
    </tr>
  </tbody>
</table>
</font>

![png](fig/clean.png)

#### _Feature Exploration_
Besides the base prices and `Volume` features, more price changing vectors and corresponding classes are derived for the proposed target and further improvement, e.g.,
`Close_pre_Close` vector: the price changes from Close of the last **pre**vious day to Close of the base day & `Close_Close_next_up` classification: the price **up**s from Close of the base day to Close of the **next** day.
Although the feature `Open_next` will limit the available time, the closest price is supposed to have the highest correlation with the target `Close_Close_next_up`.
The flatting prices are merged with upping prices, aligned with the [matplotlib.finance](http://matplotlib.org/api/finance_api.html)

<font size="1" style="line-height:11px;letter-spacing:0px">

    [Statistics of Close-to-Close (Close_Close_next) prices]
    Total number of records: 1567
    Daily prices upping:     847  (including flatting aligned w/ matplotlib.finance)
    Daily prices flatting:   1
    Daily prices downing:    720
    Percentage of daily prices upping: 54.05%
</font>    

The applied Vectors are listed below and there are also corresponding  up/down classes:
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=1>
    <tr>
      <th              >               </th>
      <th colspan = "6"> Timeline ===> </th>
    </tr>
    <tr>
      <th              >             3 Days</th>
      <th colspan = "2"> Last previous Day </th>
      <th colspan = "2">      Base     Day </th>
      <th colspan = "2">      next     Day </th>
    </tr>
    <tr>
      <th>Prices    </th>
      <th>Open_pre  </th>
      <th>Close_pre </th>
      <th>Open      </th>
      <th>Close     </th>
      <th>Open_next </th>
      <th>Close_next</th>
    </tr>
    <tr>
      <th              >              </th>
      <th colspan = "5"> X (Features) </th>
      <th              > y (Label)    </th>
    </tr>
    <tr>
      <th rowspan = "5"> Feature Vectors </th>
      <th colspan = "4">  Open_pre_Close </th>
    </tr>
    <tr>
      <th              >                 </th>
      <th colspan = "3"> Close_pre_Close </th>
    </tr>
    <tr>
      <th colspan = "2">                 </th>
      <th colspan = "2">  Open_Close     </th>
    </tr>
    <tr>
      <th colspan = "2">                 </th>
      <th colspan = "3">  Open_Open_next </th>
    </tr>
    <tr>
      <th colspan = "3">                 </th>
      <th colspan = "2"> Close_Open_next </th>
    </tr>
    <tr>
      <th rowspan = "2">  Target Vectors </th>
      <th colspan = "3">                 </th>
      <th colspan = "3"> Close_Close_next</th>
    </tr>
    <tr>
      <th colspan = "4">                 </th>
      <th colspan = "2">  Open_Close_next</th>
    </tr>
</table>
</font>

Here are the current data, the more `y` are for further discussion:
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">Base Features ~  X_base</th>
      <th colspan="5" halign="left">Vector Features ~  X_vec</th>
      <th colspan="5" halign="left">Up Features ~  X_up</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_pre_Close</th>
      <th>Close_pre_Close</th>
      <th>Open_Close</th>
      <th>Open_Open_next</th>
      <th>Close_Open_next</th>
      <th>Open_pre_Close_up</th>
      <th>Close_pre_Close_up</th>
      <th>Open_Close_up</th>
      <th>Open_Open_next_up</th>
      <th>Close_Open_next_up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>2653.83</td>
      <td>2674.72</td>
      <td>2653.83</td>
      <td>2663.99</td>
      <td>1922966393</td>
      <td>20.10</td>
      <td>21.80</td>
      <td>10.16</td>
      <td>23.07</td>
      <td>12.91</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">y (Classification)</th>
      <th colspan="2" halign="left">y (Regressions)</th>
    </tr>
    <tr>
      <th></th>
      <th>Close_Close_next_up</th>
      <th>Open_next_Close_next_up</th>
      <th>Open_next</th>
      <th>Close_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>False</td>
      <td>False</td>
      <td>2676.90</td>
      <td>2656.30</td>
    </tr>
  </tbody>
</table>
</font>

### Exploratory Visualization of Basic Data
The price group (`Open`, `High`, `Low` & `Close`) indeed have high correlations inside the group but do not help to the target Close-to-Close price change (`Close_Close_next` & `Close_Close_next_up`).
The price-change vectors have better correlations with the target price change, but the vectors including `Close_next` cannot be the feature to predict `Close_Close_next_up`.
Therefore, the best feature is up/down classified by the Close-to-next-Open vector (`Close_Open_next_up`)
![png](fig/CorrScatter.png)

![png](fig/CorrHeat.png)

### Statistics Features
The statistics are calculated by the imported [stockstats](http://pypi.python.org/pypi/stockstats)[^stats] module.
All the examples in the [Tutorial of the stockstats](http://pypi.python.org/pypi/stockstats)[^stats] are listed below and some statistics requiring multiple-day data might incomplete in the first few days:

[^stats]: ["stockstats", *PyPI - the Python Package Index*](http://pypi.python.org/pypi/stockstats).

<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volume_delta</th>
      <th>open_-2_r</th>
      <th>middle</th>
      <th>cr</th>
      <th>cr-ma1</th>
      <th>cr-ma2</th>
      <th>cr-ma3</th>
      <th>volume_-3_s</th>
      <th>volume_-1_s</th>
      <th>volume_2_s</th>
      <th>volume_-3,2,-1_max</th>
      <th>volume_-2_s</th>
      <th>volume_0_s</th>
      <th>volume_1_s</th>
      <th>volume_-3~1_min</th>
      <th>rsv_9</th>
      <th>kdjk_9</th>
      <th>kdjk</th>
      <th>kdjd_9</th>
      <th>kdjd</th>
      <th>kdjj_9</th>
      <th>kdjj</th>
      <th>open_2_sma</th>
      <th>close_26_ema</th>
      <th>macd</th>
      <th>macds</th>
      <th>macdh</th>
      <th>close_20_sma</th>
      <th>close_20_mstd</th>
      <th>boll</th>
      <th>boll_ub</th>
      <th>boll_lb</th>
      <th>cr-ma1_20_c</th>
      <th>close_-1_s</th>
      <th>close_-1_d</th>
      <th>rs_6</th>
      <th>rsi_6</th>
      <th>rs_12</th>
      <th>rsi_12</th>
      <th>wr_10</th>
      <th>wr_6</th>
      <th>middle_14_sma</th>
      <th>cci</th>
      <th>middle_20_sma</th>
      <th>cci_20</th>
      <th>tr</th>
      <th>atr</th>
      <th>close_10_sma</th>
      <th>close_50_sma</th>
      <th>dma</th>
      <th>high_delta</th>
      <th>um</th>
      <th>low_delta</th>
      <th>dm</th>
      <th>pdm</th>
      <th>pdm_14_ema</th>
      <th>pdm_14</th>
      <th>atr_14</th>
      <th>pdi_14</th>
      <th>pdi</th>
      <th>mdm</th>
      <th>mdm_14_ema</th>
      <th>mdm_14</th>
      <th>mdi_14</th>
      <th>mdi</th>
      <th>dx_14</th>
      <th>dx</th>
      <th>dx_6_ema</th>
      <th>adx</th>
      <th>adx_6_ema</th>
      <th>adxr</th>
      <th>trix</th>
      <th>trix_9_sma</th>
      <th>change</th>
      <th>vr</th>
      <th>vr_6_sma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.566000e+03</td>
      <td>1565.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1562.000000</td>
      <td>1560.000000</td>
      <td>1556.000000</td>
      <td>1.564000e+03</td>
      <td>1.566000e+03</td>
      <td>1.565000e+03</td>
      <td>1.567000e+03</td>
      <td>1.565000e+03</td>
      <td>1.567000e+03</td>
      <td>1.566000e+03</td>
      <td>1.567000e+03</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1543.0</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1567.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1566.000000</td>
      <td>1563.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.963704e+05</td>
      <td>0.096684</td>
      <td>1965.492076</td>
      <td>inf</td>
      <td>124.022463</td>
      <td>124.415433</td>
      <td>124.897290</td>
      <td>8.992478e+08</td>
      <td>9.006032e+08</td>
      <td>9.015516e+08</td>
      <td>1.039004e+09</td>
      <td>9.000359e+08</td>
      <td>9.012557e+08</td>
      <td>9.013996e+08</td>
      <td>7.662189e+08</td>
      <td>63.235438</td>
      <td>63.203835</td>
      <td>63.203835</td>
      <td>63.188361</td>
      <td>63.188361</td>
      <td>63.234785</td>
      <td>63.234785</td>
      <td>1965.043497</td>
      <td>1955.338746</td>
      <td>6.018227</td>
      <td>6.080915</td>
      <td>-0.125375</td>
      <td>1958.005085</td>
      <td>23.114686</td>
      <td>1958.005085</td>
      <td>2004.649506</td>
      <td>1912.190761</td>
      <td>20.0</td>
      <td>1965.570977</td>
      <td>0.865868</td>
      <td>inf</td>
      <td>56.986391</td>
      <td>inf</td>
      <td>56.398568</td>
      <td>36.384179</td>
      <td>38.049978</td>
      <td>1959.992878</td>
      <td>27.313742</td>
      <td>1957.453849</td>
      <td>32.060715</td>
      <td>18.605702</td>
      <td>18.306328</td>
      <td>1962.203735</td>
      <td>1944.880187</td>
      <td>17.323547</td>
      <td>0.872676</td>
      <td>4.660811</td>
      <td>0.870268</td>
      <td>4.705702</td>
      <td>4.518500</td>
      <td>4.497418</td>
      <td>4.497418</td>
      <td>18.306328</td>
      <td>25.250958</td>
      <td>25.250958</td>
      <td>4.608130</td>
      <td>4.566012</td>
      <td>4.566012</td>
      <td>23.166917</td>
      <td>23.166917</td>
      <td>31.643837</td>
      <td>31.643837</td>
      <td>31.748826</td>
      <td>31.748826</td>
      <td>31.851244</td>
      <td>31.851244</td>
      <td>0.045850</td>
      <td>0.046456</td>
      <td>0.048553</td>
      <td>inf</td>
      <td>124.972175</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.635306e+08</td>
      <td>1.072285</td>
      <td>378.076342</td>
      <td>NaN</td>
      <td>48.710956</td>
      <td>46.343472</td>
      <td>41.699368</td>
      <td>6.683190e+08</td>
      <td>6.689955e+08</td>
      <td>6.696569e+08</td>
      <td>7.498840e+08</td>
      <td>6.688323e+08</td>
      <td>6.692804e+08</td>
      <td>6.694699e+08</td>
      <td>5.740879e+08</td>
      <td>30.668103</td>
      <td>23.304411</td>
      <td>23.304411</td>
      <td>20.309376</td>
      <td>20.309376</td>
      <td>36.437987</td>
      <td>36.437987</td>
      <td>378.205704</td>
      <td>375.471750</td>
      <td>13.331871</td>
      <td>12.140937</td>
      <td>9.478684</td>
      <td>376.809832</td>
      <td>13.488080</td>
      <td>376.809832</td>
      <td>384.230382</td>
      <td>370.722225</td>
      <td>0.0</td>
      <td>377.720353</td>
      <td>15.678145</td>
      <td>NaN</td>
      <td>17.462330</td>
      <td>NaN</td>
      <td>12.123782</td>
      <td>30.541609</td>
      <td>31.084450</td>
      <td>377.313273</td>
      <td>105.830823</td>
      <td>376.931567</td>
      <td>108.398047</td>
      <td>12.441796</td>
      <td>7.122292</td>
      <td>377.397041</td>
      <td>373.423775</td>
      <td>37.929022</td>
      <td>11.890427</td>
      <td>7.150698</td>
      <td>15.066636</td>
      <td>10.052271</td>
      <td>7.186862</td>
      <td>2.179613</td>
      <td>2.179613</td>
      <td>7.122292</td>
      <td>9.468601</td>
      <td>9.468601</td>
      <td>10.072019</td>
      <td>3.900456</td>
      <td>3.900456</td>
      <td>11.717936</td>
      <td>11.717936</td>
      <td>21.137743</td>
      <td>21.137743</td>
      <td>15.385488</td>
      <td>15.385488</td>
      <td>13.481207</td>
      <td>13.481207</td>
      <td>0.107835</td>
      <td>0.103251</td>
      <td>0.789631</td>
      <td>NaN</td>
      <td>52.052547</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.854808e+09</td>
      <td>-6.911553</td>
      <td>1275.823333</td>
      <td>35.036925</td>
      <td>44.631415</td>
      <td>52.251135</td>
      <td>55.682846</td>
      <td>1.839316e+08</td>
      <td>1.839316e+08</td>
      <td>1.839316e+08</td>
      <td>3.782322e+08</td>
      <td>1.839316e+08</td>
      <td>1.839316e+08</td>
      <td>1.839316e+08</td>
      <td>1.839316e+08</td>
      <td>0.000000</td>
      <td>4.666604</td>
      <td>4.666604</td>
      <td>9.879499</td>
      <td>9.879499</td>
      <td>-22.128273</td>
      <td>-22.128273</td>
      <td>1278.055000</td>
      <td>1308.040000</td>
      <td>-47.639949</td>
      <td>-40.305676</td>
      <td>-61.144555</td>
      <td>1308.040000</td>
      <td>3.228371</td>
      <td>1308.040000</td>
      <td>1320.170742</td>
      <td>1270.300916</td>
      <td>20.0</td>
      <td>1278.040000</td>
      <td>-113.190000</td>
      <td>0.067665</td>
      <td>6.337629</td>
      <td>0.174513</td>
      <td>14.858333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1302.380000</td>
      <td>-340.578134</td>
      <td>1302.380000</td>
      <td>-395.599439</td>
      <td>3.700000</td>
      <td>6.807037</td>
      <td>1305.426000</td>
      <td>1308.040000</td>
      <td>-125.243000</td>
      <td>-72.360000</td>
      <td>0.000000</td>
      <td>-121.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.807037</td>
      <td>4.532342</td>
      <td>4.532342</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010822</td>
      <td>0.010822</td>
      <td>6.959956</td>
      <td>6.959956</td>
      <td>10.029961</td>
      <td>10.029961</td>
      <td>-0.368495</td>
      <td>-0.346507</td>
      <td>-4.097924</td>
      <td>34.691762</td>
      <td>46.627682</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.638344e+07</td>
      <td>-0.414442</td>
      <td>1677.423333</td>
      <td>88.676661</td>
      <td>90.856580</td>
      <td>91.869615</td>
      <td>94.586037</td>
      <td>4.980904e+08</td>
      <td>4.981410e+08</td>
      <td>4.980993e+08</td>
      <td>5.645439e+08</td>
      <td>4.980993e+08</td>
      <td>4.981828e+08</td>
      <td>4.981410e+08</td>
      <td>4.447109e+08</td>
      <td>39.321951</td>
      <td>44.265535</td>
      <td>44.265535</td>
      <td>47.315250</td>
      <td>47.315250</td>
      <td>33.997150</td>
      <td>33.997150</td>
      <td>1673.200000</td>
      <td>1663.401132</td>
      <td>-0.729602</td>
      <td>-0.239761</td>
      <td>-5.156739</td>
      <td>1662.220500</td>
      <td>14.346560</td>
      <td>1662.220500</td>
      <td>1709.842242</td>
      <td>1616.332248</td>
      <td>20.0</td>
      <td>1676.155000</td>
      <td>-5.750000</td>
      <td>0.817651</td>
      <td>44.983924</td>
      <td>0.921875</td>
      <td>47.967473</td>
      <td>8.607483</td>
      <td>9.642646</td>
      <td>1664.025238</td>
      <td>-45.143013</td>
      <td>1661.656750</td>
      <td>-41.853124</td>
      <td>10.762500</td>
      <td>13.804070</td>
      <td>1666.865000</td>
      <td>1651.988100</td>
      <td>-2.196300</td>
      <td>-4.580000</td>
      <td>0.000000</td>
      <td>-5.690000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.973251</td>
      <td>2.973251</td>
      <td>13.804070</td>
      <td>18.420441</td>
      <td>18.420441</td>
      <td>0.000000</td>
      <td>2.237666</td>
      <td>2.237666</td>
      <td>14.659956</td>
      <td>14.659956</td>
      <td>14.102412</td>
      <td>14.102412</td>
      <td>20.190743</td>
      <td>20.190743</td>
      <td>21.837309</td>
      <td>21.837309</td>
      <td>-0.013717</td>
      <td>-0.011945</td>
      <td>-0.303388</td>
      <td>85.762681</td>
      <td>87.429520</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.064320e+05</td>
      <td>0.141623</td>
      <td>2003.896667</td>
      <td>115.941285</td>
      <td>117.850998</td>
      <td>117.699145</td>
      <td>118.016859</td>
      <td>5.784377e+08</td>
      <td>5.786119e+08</td>
      <td>5.785919e+08</td>
      <td>6.639878e+08</td>
      <td>5.785919e+08</td>
      <td>5.786320e+08</td>
      <td>5.786119e+08</td>
      <td>5.028454e+08</td>
      <td>70.539419</td>
      <td>67.622746</td>
      <td>67.622746</td>
      <td>66.364005</td>
      <td>66.364005</td>
      <td>69.629701</td>
      <td>69.629701</td>
      <td>2003.870000</td>
      <td>1990.847683</td>
      <td>7.656907</td>
      <td>7.336798</td>
      <td>-0.045354</td>
      <td>1996.246000</td>
      <td>20.022851</td>
      <td>1996.246000</td>
      <td>2076.214072</td>
      <td>1947.083331</td>
      <td>20.0</td>
      <td>2002.305000</td>
      <td>0.875000</td>
      <td>1.398057</td>
      <td>58.299576</td>
      <td>1.356655</td>
      <td>57.566972</td>
      <td>29.109762</td>
      <td>30.852995</td>
      <td>1997.535238</td>
      <td>53.353845</td>
      <td>1995.207833</td>
      <td>58.326824</td>
      <td>15.260000</td>
      <td>16.409707</td>
      <td>1999.859000</td>
      <td>1986.981200</td>
      <td>21.680600</td>
      <td>0.890000</td>
      <td>0.890000</td>
      <td>1.935000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.063428</td>
      <td>4.063428</td>
      <td>16.409707</td>
      <td>24.910335</td>
      <td>24.910335</td>
      <td>0.000000</td>
      <td>3.482782</td>
      <td>3.482782</td>
      <td>21.313218</td>
      <td>21.313218</td>
      <td>29.490164</td>
      <td>29.490164</td>
      <td>28.054626</td>
      <td>28.054626</td>
      <td>28.761045</td>
      <td>28.761045</td>
      <td>0.060143</td>
      <td>0.060233</td>
      <td>0.048393</td>
      <td>111.021171</td>
      <td>112.594169</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.965710e+07</td>
      <td>0.678805</td>
      <td>2164.215000</td>
      <td>145.610277</td>
      <td>145.149435</td>
      <td>144.404614</td>
      <td>147.860193</td>
      <td>8.610650e+08</td>
      <td>8.647954e+08</td>
      <td>8.658505e+08</td>
      <td>1.317360e+09</td>
      <td>8.627076e+08</td>
      <td>8.656709e+08</td>
      <td>8.657607e+08</td>
      <td>6.326820e+08</td>
      <td>91.128403</td>
      <td>83.316911</td>
      <td>83.316911</td>
      <td>80.128939</td>
      <td>80.128939</td>
      <td>95.312446</td>
      <td>95.312446</td>
      <td>2164.012500</td>
      <td>2155.133071</td>
      <td>14.557992</td>
      <td>13.711731</td>
      <td>5.380283</td>
      <td>2153.926500</td>
      <td>27.256386</td>
      <td>2153.926500</td>
      <td>2193.460103</td>
      <td>2118.896085</td>
      <td>20.0</td>
      <td>2164.237500</td>
      <td>8.897500</td>
      <td>2.400138</td>
      <td>70.589428</td>
      <td>1.864360</td>
      <td>65.088190</td>
      <td>59.784172</td>
      <td>63.018686</td>
      <td>2155.529405</td>
      <td>104.370010</td>
      <td>2153.903167</td>
      <td>111.178263</td>
      <td>22.452500</td>
      <td>19.598908</td>
      <td>2159.268000</td>
      <td>2149.199700</td>
      <td>42.429600</td>
      <td>7.107500</td>
      <td>7.107500</td>
      <td>8.587500</td>
      <td>5.690000</td>
      <td>6.945000</td>
      <td>5.549610</td>
      <td>5.549610</td>
      <td>19.598908</td>
      <td>31.628644</td>
      <td>31.628644</td>
      <td>5.440000</td>
      <td>5.550190</td>
      <td>5.550190</td>
      <td>29.817200</td>
      <td>29.817200</td>
      <td>45.746425</td>
      <td>45.746425</td>
      <td>41.010204</td>
      <td>41.010204</td>
      <td>39.850599</td>
      <td>39.850599</td>
      <td>0.119158</td>
      <td>0.117629</td>
      <td>0.467055</td>
      <td>150.781448</td>
      <td>149.848066</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.765587e+09</td>
      <td>6.050461</td>
      <td>2863.973333</td>
      <td>inf</td>
      <td>449.383886</td>
      <td>449.383886</td>
      <td>449.383886</td>
      <td>4.024144e+09</td>
      <td>4.024144e+09</td>
      <td>4.024144e+09</td>
      <td>4.024144e+09</td>
      <td>4.024144e+09</td>
      <td>4.024144e+09</td>
      <td>4.024144e+09</td>
      <td>2.838299e+09</td>
      <td>100.000000</td>
      <td>97.792533</td>
      <td>97.792533</td>
      <td>96.055363</td>
      <td>96.055363</td>
      <td>123.515157</td>
      <td>123.515157</td>
      <td>2857.355000</td>
      <td>2781.595585</td>
      <td>47.662717</td>
      <td>43.182659</td>
      <td>27.413893</td>
      <td>2801.856500</td>
      <td>91.627922</td>
      <td>2801.856500</td>
      <td>2945.158618</td>
      <td>2722.922774</td>
      <td>20.0</td>
      <td>2872.870000</td>
      <td>72.890000</td>
      <td>inf</td>
      <td>100.000000</td>
      <td>inf</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>2821.722381</td>
      <td>318.649509</td>
      <td>2801.216333</td>
      <td>278.083764</td>
      <td>125.220000</td>
      <td>50.759978</td>
      <td>2835.381000</td>
      <td>2748.848600</td>
      <td>129.493200</td>
      <td>46.510000</td>
      <td>46.510000</td>
      <td>89.760000</td>
      <td>121.800000</td>
      <td>46.510000</td>
      <td>14.881184</td>
      <td>14.881184</td>
      <td>50.759978</td>
      <td>64.384652</td>
      <td>64.384652</td>
      <td>121.800000</td>
      <td>36.356521</td>
      <td>36.356521</td>
      <td>95.970924</td>
      <td>95.970924</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>0.264922</td>
      <td>0.252039</td>
      <td>3.902828</td>
      <td>inf</td>
      <td>422.185018</td>
    </tr>
  </tbody>
</table>
</font>

The statistics with the most day number of data:
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Statistics</th>
      <th>Days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>close_50_sma</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>close_26_ema</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close_20_sma</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>close_20_mstd</td>
      <td>20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cr-ma1_20_c</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</font>

#### _Feature Comparison_
The first and last 50 days and constant statistics will be dropped to guarantee the integrity.
In the Top 10 Positive/Negative Correlation with `close_close_next_up`/`close_close_next` (the [stockstats](http://pypi.python.org/pypi/stockstats)[^stats] changes all column names to lower case), the best statistics features are 6/12 days Relative Strength Index (RSI) and 6/10 days Williams Overbought/Oversold Index (WR):

<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Positive Correlation</th>
      <th colspan="3" halign="left">Negitive Correlation</th>
    </tr>
    <tr>
      <th></th>
      <th>Features</th>
      <th>close_close_next_up</th>
      <th>close_close_next</th>
      <th>Features</th>
      <th>close_close_next_up</th>
      <th>close_close_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>close_close_next_up</td>
      <td>100.00%</td>
      <td>71.51%</td>
      <td>open_close_up</td>
      <td>-7.82%</td>
      <td>-4.62%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>open_next_close_next_up</td>
      <td>86.01%</td>
      <td>69.66%</td>
      <td>close_pre_close_up</td>
      <td>-6.62%</td>
      <td>-3.58%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close_close_next</td>
      <td>71.51%</td>
      <td>100.00%</td>
      <td>rsi_12</td>
      <td>-5.60%</td>
      <td>-6.59%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>open_next_close_next</td>
      <td>69.33%</td>
      <td>97.70%</td>
      <td>open_close</td>
      <td>-5.59%</td>
      <td>-1.63%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>close_open_next</td>
      <td>36.76%</td>
      <td>48.33%</td>
      <td>close_-1_d</td>
      <td>-5.53%</td>
      <td>-1.75%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>close_open_next_up</td>
      <td>35.68%</td>
      <td>40.68%</td>
      <td>close_pre_close</td>
      <td>-5.53%</td>
      <td>-1.75%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>volume</td>
      <td>5.51%</td>
      <td>5.52%</td>
      <td>rsv_9</td>
      <td>-5.44%</td>
      <td>-6.35%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>volume_0_s</td>
      <td>5.51%</td>
      <td>5.52%</td>
      <td>rsi_6</td>
      <td>-5.43%</td>
      <td>-6.33%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wr_6</td>
      <td>5.39%</td>
      <td>6.36%</td>
      <td>rs_6</td>
      <td>-5.30%</td>
      <td>-3.57%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>wr_10</td>
      <td>5.33%</td>
      <td>6.59%</td>
      <td>change</td>
      <td>-5.04%</td>
      <td>-1.50%</td>
    </tr>
  </tbody>
</table>
</font>

#### _Feature Cleaning & Selection_
Selecting and setup the most correlated RSI6/12, WR6/10 and the popular rolling means (2 days simple moving average, C2M), Moving Average Convergence Divergence (MACD) and Bollinger Bands (Boll/u/l) suggested by proposal comment.
The first 11 days without sufficient data for 12-day rsi_12 should be dropped as usual.

<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RSI12</th>
      <th>RSI6</th>
      <th>WR10</th>
      <th>WR6</th>
      <th>C2M</th>
      <th>MACD</th>
      <th>Boll_u</th>
      <th>Boll</th>
      <th>Boll_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-01-19</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.408879</td>
      <td>0.408879</td>
      <td>1308.040</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1308.040000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-01-20</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>4.040816</td>
      <td>4.040816</td>
      <td>1311.270</td>
      <td>0.144936</td>
      <td>1320.405820</td>
      <td>1311.270000</td>
      <td>1302.134180</td>
    </tr>
  </tbody>
</table>
</font>

#### _Correlation of Current Data_
<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Positive Correlation</th>
      <th colspan="3" halign="left">Negitive Correlation</th>
    </tr>
    <tr>
      <th></th>
      <th>Features</th>
      <th>Close_Close_next_up</th>
      <th>Close_Close_next</th>
      <th>Features</th>
      <th>Close_Close_next_up</th>
      <th>Close_Close_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Close_Close_next_up</td>
      <td>100.00%</td>
      <td>69.07%</td>
      <td>Open_Close_up</td>
      <td>-6.73%</td>
      <td>-3.63%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Open_next_Close_next_up</td>
      <td>86.13%</td>
      <td>67.29%</td>
      <td>RSI12</td>
      <td>-5.81%</td>
      <td>-6.38%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Close_Close_next</td>
      <td>69.07%</td>
      <td>100.00%</td>
      <td>Close_pre_Close_up</td>
      <td>-5.75%</td>
      <td>-1.69%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Open_next_Close_next</td>
      <td>66.76%</td>
      <td>96.90%</td>
      <td>Close_pre_Close</td>
      <td>-5.65%</td>
      <td>-2.12%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Close_Open_next_up</td>
      <td>34.48%</td>
      <td>37.33%</td>
      <td>RSI6</td>
      <td>-5.64%</td>
      <td>-5.58%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Close_Open_next</td>
      <td>30.96%</td>
      <td>43.95%</td>
      <td>MACD</td>
      <td>-5.23%</td>
      <td>-7.34%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>WR10</td>
      <td>5.50%</td>
      <td>5.48%</td>
      <td>Open_Close</td>
      <td>-5.12%</td>
      <td>-1.86%</td>
    </tr>
  </tbody>
</table>
</font>

### Exploratory Visualization of All Data
Checking data by stick plots which including all base features (`Open`, `High`, `Low`, `Close` and `Volume`) and Bollinger bands and zooming in the test data
![png](fig/StickAll.png)

![png](fig/StickTest.png)

Checking individual data and zoom in the test data according to scale groups of Prices, Price Changes and Indices

![png](fig/Price.png)
![png](fig/PriceChange.png)
![png](fig/Index.png)

## Algorithms and Techniques
### Target Model
- [**Ensemble Tree Gradient Boosting Classifier**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC]
    - Application: Ranking webs for the commercial search engines, e.g., Yahoo and Yandex (Ref.: [`wikipedia/Gradient_boosting`](https://wikipedia.org/wiki/Gradient_boosting))
    - Pros (Ref.: [`sklearn/ensemble#gradient-tree-boosting`](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting))
        - Natural handling of mixed-type data (heterogeneous features)
        - High predictive power
        - Robustness to outliers in output space (via robust loss functions)
        - Fast training & prediction (based on the following experiment)
    - Cons (Ref.: [`sklearn/ensemble#gradient-tree-boosting`](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting))
        - Scalability, due to the sequential nature of boosting it can hardly be parallelized
    - Natural handling of mixed-type data (heterogeneous features) and more powerful for classification when the number of samples < 100K (Ref.: [`sklearn/ensemble#gradient-tree-boosting`](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) & [`machine_learning_map`](http://scikit-learn.org/stable/tutorial/machine_learning_map))
    
### Benchmark Model
- [**Support Vector Machines**](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[^svc]
    - Application: Text and hypertext categorization (Ref.: [`wikipedia/Support_vector_machine`](https://wikipedia.org/wiki/Support_vector_machine))
    - Pros (Ref.: [`sklearn/svm`](http://scikit-learn.org/stable/modules/svm.html))
        - Effective in high dimensional spaces (even when the number of dimensions is greater than the number of samples)
        - Memory efficient (use some training points for the decision function ~ support vectors)
        - Versatile in the decision function (common and custom kernel functions)
    - Cons (Ref.: [`youtube/Udacity/SVM Strengths and Weaknesses`](https://youtu.be/U9-ZsbaaGAs) & [`sklearn/svm`](http://scikit-learn.org/stable/modules/svm.html))
        - When the number of features is much greater than the number of samples, need specified kernel function and regularization to avoid over-fitting
        - Do not directly provide probability estimates
        - High computation cost when training large data
        - Low noise/overlapping tolerance
    - Efficient for classification when the number of samples < 100K (Ref.: [`sklearn/machine_learning_map`](http://scikit-learn.org/stable/tutorial/machine_learning_map))

The selected benchmark model will be trained and tested in parallel with the target solution model.

[^k]: ["sklearn.neighbors.KNeighborsClassifier," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

[^svc]: ["sklearn.svm.SVC," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

[^naïve]: [udacity, "Project: Finding Donors for CharityML," *github.com*](http://github.com/udacity/machine-learning/blob/master/projects/finding_donors/finding_donors.ipynb)

## III. Methodology

### Data Preprocessing
#### _Log-Transforming the Skewed Continuous Feature_
![png](fig/Log.png)

#### _Normalizing Numerical Features_
Log-transformed data with MinMaxScaler is referred because it seems the most normal and cleanest

<font size="1" style="line-height:11px;letter-spacing:0px">

    [data_log with MinMaxScaler]   
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_pre</th>
      <th>Close_pre</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_next</th>
      <th>Close_next</th>
      <th>Open_pre_Close</th>
      <th>...</th>
      <th>Open_next_Close_next_up</th>
      <th>RSI12</th>
      <th>RSI6</th>
      <th>WR10</th>
      <th>WR6</th>
      <th>C2M</th>
      <th>MACD</th>
      <th>Boll_u</th>
      <th>Boll</th>
      <th>Boll_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>0.859482</td>
      <td>0.855358</td>
      <td>0.865736</td>
      <td>0.875402</td>
      <td>0.875279</td>
      <td>0.869027</td>
      <td>0.760673</td>
      <td>0.880251</td>
      <td>0.864205</td>
      <td>0.655098</td>
      <td>...</td>
      <td>False</td>
      <td>0.490706</td>
      <td>0.589765</td>
      <td>0.088736</td>
      <td>0.121311</td>
      <td>0.867446</td>
      <td>0.294031</td>
      <td>0.881959</td>
      <td>0.900657</td>
      <td>0.882352</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_pre</th>
      <th>Close_pre</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_next</th>
      <th>Close_next</th>
      <th>Open_pre_Close</th>
      <th>...</th>
      <th>Open_next_Close_next</th>
      <th>RSI12</th>
      <th>RSI6</th>
      <th>WR10</th>
      <th>WR6</th>
      <th>C2M</th>
      <th>MACD</th>
      <th>Boll_u</th>
      <th>Boll</th>
      <th>Boll_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>...</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
      <td>1555.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.435273</td>
      <td>0.433980</td>
      <td>0.435811</td>
      <td>0.437971</td>
      <td>0.438365</td>
      <td>0.434520</td>
      <td>0.449689</td>
      <td>0.436357</td>
      <td>0.435048</td>
      <td>0.586991</td>
      <td>...</td>
      <td>0.566837</td>
      <td>0.567807</td>
      <td>0.587483</td>
      <td>0.364840</td>
      <td>0.381178</td>
      <td>0.436874</td>
      <td>0.563489</td>
      <td>0.417812</td>
      <td>0.437382</td>
      <td>0.444838</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.236146</td>
      <td>0.235178</td>
      <td>0.236173</td>
      <td>0.236469</td>
      <td>0.236310</td>
      <td>0.235213</td>
      <td>0.191283</td>
      <td>0.236218</td>
      <td>0.235254</td>
      <td>0.076895</td>
      <td>...</td>
      <td>0.078219</td>
      <td>0.163871</td>
      <td>0.202026</td>
      <td>0.306013</td>
      <td>0.311222</td>
      <td>0.236589</td>
      <td>0.140330</td>
      <td>0.237235</td>
      <td>0.250816</td>
      <td>0.253671</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</font>

#### _Data Preprocessing_
Original Scaled/Normalized Features and 29-day previous-data-concatenated Features are shown below and the best day range will be tried later

<font size="1" style="line-height:11px;letter-spacing:0px">

    Original Scaled/Normalized Features:
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">Base Features ~  X_base</th>
      <th colspan="5" halign="left">Vector Features ~  X_vec</th>
      <th colspan="5" halign="left">Up Features ~  X_up</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_pre_Close</th>
      <th>Close_pre_Close</th>
      <th>Open_Close</th>
      <th>Open_Open_next</th>
      <th>Close_Open_next</th>
      <th>Open_pre_Close_up</th>
      <th>Close_pre_Close_up</th>
      <th>Open_Close_up</th>
      <th>Open_Open_next_up</th>
      <th>Close_Open_next_up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>0.865736</td>
      <td>0.875402</td>
      <td>0.875279</td>
      <td>0.869027</td>
      <td>0.760673</td>
      <td>0.655098</td>
      <td>0.725441</td>
      <td>0.619244</td>
      <td>0.737713</td>
      <td>0.721379</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
    29-day Previous-data-concated Features:
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_pre_Close</th>
      <th>Close_pre_Close</th>
      <th>Open_Close</th>
      <th>Open_Open_next</th>
      <th>Close_Open_next</th>
      <th>...</th>
      <th>Close_Open_next_up_pre29</th>
      <th>RSI12_pre29</th>
      <th>RSI6_pre29</th>
      <th>WR10_pre29</th>
      <th>WR6_pre29</th>
      <th>C2M_pre29</th>
      <th>MACD_pre29</th>
      <th>Boll_u_pre29</th>
      <th>Boll_pre29</th>
      <th>Boll_l_pre29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>0.865736</td>
      <td>0.875402</td>
      <td>0.875279</td>
      <td>0.869027</td>
      <td>0.760673</td>
      <td>0.655098</td>
      <td>0.725441</td>
      <td>0.619244</td>
      <td>0.737713</td>
      <td>0.721379</td>
      <td>...</td>
      <td>False</td>
      <td>0.381866</td>
      <td>0.352470</td>
      <td>0.860849</td>
      <td>0.860849</td>
      <td>0.894359</td>
      <td>0.437267</td>
      <td>0.919090</td>
      <td>0.934223</td>
      <td>0.910176</td>
    </tr>
  </tbody>
</table>
</font>

#### _Split Data_
Split the data (both features and their labels) into training and test sets.
Data before 2018 will be used for training and the other for testing.

<font size="1" style="line-height:11px;letter-spacing:0px">

    Training set has 1456 samples, tail:
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Base Features ~  X_base</th>
      <th>...</th>
      <th colspan="10" halign="left">Statistics Features ~  X_stat</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_pre1</th>
      <th>High_pre1</th>
      <th>Low_pre1</th>
      <th>Close_pre1</th>
      <th>Volume_pre1</th>
      <th>...</th>
      <th>Boll_l_pre28</th>
      <th>RSI12_pre29</th>
      <th>RSI6_pre29</th>
      <th>WR10_pre29</th>
      <th>WR6_pre29</th>
      <th>C2M_pre29</th>
      <th>MACD_pre29</th>
      <th>Boll_u_pre29</th>
      <th>Boll_pre29</th>
      <th>Boll_l_pre29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-30</th>
      <td>0.887958</td>
      <td>0.886344</td>
      <td>0.887761</td>
      <td>0.875059</td>
      <td>0.647104</td>
      <td>0.886039</td>
      <td>0.883539</td>
      <td>0.893490</td>
      <td>0.883793</td>
      <td>0.588721</td>
      <td>...</td>
      <td>0.885936</td>
      <td>0.597805</td>
      <td>0.581406</td>
      <td>0.287592</td>
      <td>0.136336</td>
      <td>0.818263</td>
      <td>0.608090</td>
      <td>0.784622</td>
      <td>0.849869</td>
      <td>0.885784</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">y (Classification)</th>
      <th colspan="2" halign="left">y (Regressions)</th>
    </tr>
    <tr>
      <th></th>
      <th>Close_Close_next_up</th>
      <th>Open_next_Close_next_up</th>
      <th>Open_next</th>
      <th>Close_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-30</th>
      <td>True</td>
      <td>True</td>
      <td>0.884548</td>
      <td>0.888979</td>
    </tr>
  </tbody>
</table>

    The Date to Split:  01 Jan 2018    
    Testing set has 70 samples, head:
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Base Features ~  X_base</th>
      <th>...</th>
      <th colspan="10" halign="left">Statistics Features ~  X_stat</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Open_pre1</th>
      <th>High_pre1</th>
      <th>Low_pre1</th>
      <th>Close_pre1</th>
      <th>Volume_pre1</th>
      <th>...</th>
      <th>Boll_l_pre28</th>
      <th>RSI12_pre29</th>
      <th>RSI6_pre29</th>
      <th>WR10_pre29</th>
      <th>WR6_pre29</th>
      <th>C2M_pre29</th>
      <th>MACD_pre29</th>
      <th>Boll_u_pre29</th>
      <th>Boll_pre29</th>
      <th>Boll_l_pre29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>0.865736</td>
      <td>0.875402</td>
      <td>0.875279</td>
      <td>0.869027</td>
      <td>0.760673</td>
      <td>0.859482</td>
      <td>0.867046</td>
      <td>0.866079</td>
      <td>0.855358</td>
      <td>0.737184</td>
      <td>...</td>
      <td>0.915906</td>
      <td>0.381866</td>
      <td>0.352470</td>
      <td>0.860849</td>
      <td>0.860849</td>
      <td>0.894359</td>
      <td>0.437267</td>
      <td>0.919090</td>
      <td>0.934223</td>
      <td>0.910176</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">y (Classification)</th>
      <th colspan="2" halign="left">y (Regressions)</th>
    </tr>
    <tr>
      <th></th>
      <th>Close_Close_next_up</th>
      <th>Open_next_Close_next_up</th>
      <th>Open_next</th>
      <th>Close_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-13</th>
      <td>False</td>
      <td>False</td>
      <td>0.880251</td>
      <td>0.864205</td>
    </tr>
  </tbody>
</table>
</font>

### Implementation
#### _Initial Model Evaluation_
Using the default settings and fixed `random_state` for each model.
Applying originally proposed base features (`Open`, `High`, `Low`, `Close`, `Volume`) from the raw data and trying the concatenated previous features to 29 days (totally 30 days).
The confusion matrix and classification report are clear to show that the predictions are **always up**.
The reason might be **the prices in 2018 are usually higher than previous years** even after normalization.
Therefore, the **relative price change Vectors** will be involved besides the **absolute prices**.

<font size="1" style="line-height:11px;letter-spacing:0px">

    The best classifier is SVC with 71.56% f1-score and 1-day features to predict Close_Close_next_up 
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Up_predict</th>
      <th>Down_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Up_true</th>
      <td>39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Down_true</th>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

                 precision    recall  f1-score   support
             Up       0.56      1.00      0.72        39
           Down       0.00      0.00      0.00        31
    avg / total       0.31      0.56      0.40        70
</font>

![png](fig/BaseAcc.png)

![png](fig/BaseF1.png)

The previous stick plot shows that the prices in 2018 usually higher than previous years.
![png](fig/StickAll.png)

### Refinement
#### _Advanced Features_
Besides the previous base features, all researched features above are applied here.
The classifier trained with 8-day features has great improvement while too long days with weak correlations cause too much overfitting.
Therefore, the next tuning will use the same 8-day features to tune the hyperparameters to reduce the **overfitting**.

<font size="1" style="line-height:11px;letter-spacing:0px">

    The best classifier is GradientBoostingClassifier with 75.61% f1-score and 8-day features to predict Close_Close_next_up 
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Up_predict</th>
      <th>Down_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Up_true</th>
      <td>31</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Down_true</th>
      <td>12</td>
      <td>19</td>
    </tr>
  </tbody>
</table>

                 precision    recall  f1-score   support
             Up       0.72      0.79      0.76        39
           Down       0.70      0.61      0.66        31
    avg / total       0.71      0.71      0.71        70
</font>

![png](fig/AllAcc.png)
![png](fig/AllF1.png)

#### _Feature Importance_
The best feature is the vector `Close_Open_next` as expected. The importances of the `GradientBoostingRegressor` are also listed by the way for reference.

<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Classifier</th>
      <th colspan="2" halign="left">Regressor</th>
    </tr>
    <tr>
      <th></th>
      <th>Features</th>
      <th>Importances</th>
      <th>Features</th>
      <th>Importances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Close_Open_next</td>
      <td>15.74%</td>
      <td>Close_Open_next</td>
      <td>13.55%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Close_Open_next_pre1</td>
      <td>3.25%</td>
      <td>Close_Open_next_pre1</td>
      <td>3.24%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Volume_pre5</td>
      <td>2.82%</td>
      <td>Volume_pre1</td>
      <td>2.60%</td>
    </tr>
  </tbody>
</table>
</font>

#### _Model Tuning_
Based on the 8-day features, tuning the key hyperparameters of the [Ensemble Tree Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)[^GBC] by [Exhaustive Grid Search](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)[^hyper] with [Cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html)[^cv] and [TimeSeriesSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)[^time] to overcome the **overfitting**.

[^hyper]: ["Tuning the hyper-parameters of an estimator," *scikit-learn.org*](http://scikit-learn.org/stable/modules/grid_search.html)

[^cv]: ["Cross-validation: evaluating estimator performance," *scikit-learn.org*](http://scikit-learn.org/stable/modules/cross_validation.html)

[^time]: ["sklearn.model_selection.TimeSeriesSplit," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

Wide-range hyperparameters has been tested: `learning_rate` (0.005~0.2), `n_estimators` (20~110), `max_depth` (2~16), `min_samples_split` (2~15), `min_samples_leaf` (1~8), `max_features` (0.1~None) and `subsample` (0.6~1).
However, the huge combinations need to be partitioned into many [steps](http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python)[^GBM_param] to reduce the time complexity.
The detail ranges are in the notebooks and the sample is demonstrated below.
The overfitting is easy to overcome but, the required [$F_1-score$](http://wikipedia.org/wiki/F1_score)[^f1] is improved minor and losing a little accuracy.

[^GBM_param]: [AARSHAY JAIN, "Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python," *analyticsvidhya.com*](http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python)

<font size="1" style="line-height:11px;letter-spacing:0px">
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="7" halign="left">Parameter Grid</th>
    </tr>
    <tr>
      <th></th>
      <th>learning_rate</th>
      <th>n_estimators</th>
      <th>max_depth</th>
      <th>min_samples_split</th>
      <th>min_samples_leaf</th>
      <th>max_features</th>
      <th>subsample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.03</td>
      <td>50</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>sqrt</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.04</td>
      <td>100</td>
      <td>3</td>
      <td>8</td>
      <td>2</td>
      <td></td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.05</td>
      <td></td>
      <td></td>
      <td>9</td>
      <td></td>
      <td></td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>

    Fitting 3 folds for each of 216 candidates, totalling 648 fits    
    [Parallel(n_jobs=1)]: Done 648 out of 648 | elapsed:  1.4min finished

<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Train Accuracy</th>
      <th>Test Accuracy</th>
      <th>Train f1</th>
      <th>Test f1</th>
      <th>learning_rate</th>
      <th>n_estimators</th>
      <th>max_depth</th>
      <th>min_samples_split</th>
      <th>min_samples_leaf</th>
      <th>max_features</th>
      <th>subsample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Default Model</th>
      <td>87.0%</td>
      <td>71.4%</td>
      <td>88.0%</td>
      <td>75.6%</td>
      <td>0.10</td>
      <td>100</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>None</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Optimized Model</th>
      <td>71.2%</td>
      <td>68.6%</td>
      <td>75.4%</td>
      <td>76.1%</td>
      <td>0.04</td>
      <td>50</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>sqrt</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Default</th>
      <th colspan="2" halign="left">Optimized</th>
    </tr>
    <tr>
      <th></th>
      <th>Features</th>
      <th>Importances</th>
      <th>Features</th>
      <th>Importances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Close_Open_next</td>
      <td>15.74%</td>
      <td>Close_Open_next</td>
      <td>12.16%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Close_Open_next_pre1</td>
      <td>3.25%</td>
      <td>Close_Open_next_up</td>
      <td>9.78%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Volume_pre5</td>
      <td>2.82%</td>
      <td>RSI6</td>
      <td>3.86%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume_pre1</td>
      <td>2.63%</td>
      <td>Open_Open_next_pre6</td>
      <td>3.06%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RSI6</td>
      <td>2.35%</td>
      <td>WR10_pre6</td>
      <td>2.60%</td>
    </tr>
  </tbody>
</table>
<table border=0>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Default</th>
      <th colspan="2" halign="left">Optimized</th>
    </tr>
    <tr>
      <th></th>
      <th>Up_predict</th>
      <th>Down_predict</th>
      <th>Up_predict</th>
      <th>Down_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Up_true</th>
      <td>31</td>
      <td>8</td>
      <td>35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Down_true</th>
      <td>12</td>
      <td>19</td>
      <td>18</td>
      <td>13</td>
    </tr>
  </tbody>
</table>

                 precision    recall  f1-score   support
             Up       0.66      0.90      0.76        39
           Down       0.76      0.42      0.54        31
    avg / total       0.71      0.69      0.66        70
</font>

#### _Feature Selection_
Based on the high feature importances and correlations above, `Volume` (Base Feature), `WR10`, `RSI6` (Statistics Features), Vector and corresponding Up Features are selected to explore huge feature combinations for improvement of overfitting and accuracy at the same time.
Finally, the [$F_1-score$](http://wikipedia.org/wiki/F1_score)[^f1] can be improved to 83.72%.

<font size="1" style="line-height:11px;letter-spacing:0px">

     14%|██████████▌  | 235/1716 [30:01<2:52:33,  6.99s/it]
    83.72% f1 by 12-day Volume, Open_pre_Close, Close_pre_Close_up, Open_Close_up, Close_Open_next & WR10
</font>

## IV. Results

### Model Evaluation, Validation, Justification and Visualization
Based on the features above and wide-range hyperparameters tested, the best result tested with the unseen data this year has very near training/testing scores that are quite reasonable, trusted and good than expectation and the [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[^svc] benchmark model, although the overfitting still can be improved and the testing set is a little small.
The model is robust to the incoming data everyday, e.g., the [$F_1-score$](http://wikipedia.org/wiki/F1_score)[^f1] is improved from 83.72% to 84.09% with the last coming data of 2018-04-17 (comparing the notebooks Stock_Up_Mincent_0414.ipynb and Stock_Up_Mincent_0417.ipynb).
The solution should be enough for the defined problem and conditions (only daily prices and volume features) currently, but for practical applications, the model should be re-trained continuously with the latest incoming data to learn the latest evolution of the market behavior.

<font size="1" style="line-height:11px;letter-spacing:0px">

    The best classifier is GradientBoostingClassifier with 83.72% f1-score and 12-day features to predict Close_Close_next_up 
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Up_predict</th>
      <th>Down_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Up_true</th>
      <td>36</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Down_true</th>
      <td>11</td>
      <td>20</td>
    </tr>
  </tbody>
</table>

                 precision    recall  f1-score   support
             Up       0.77      0.92      0.84        39
           Down       0.87      0.65      0.74        31
    avg / total       0.81      0.80      0.79        70
</font>

![png](fig/SelAcc.png)

![png](fig/SelF1.png)

## V. Conclusion
The best significant visualization of this project is the latest high score plotting.
### Reflection
#### _Process Summary_
- Data Engineering
  - Data Getting
  - Data Cleaning
- Feature Engineering
  - **Deriving** Statistics, Vector and Corresponding Classification Features and Labels
  - **Feature Selection** by Visualization and Comparison of Data Correlations
  - Log-Transforming the Skewed Continuous Feature
  - Normalizing Numerical Features
  - Feature Preprocessing for Stacking Daily Data of Day Range
  - Splitting Data for Training and Testing
- Model Tuning
  - Initial Model Evaluation
  - Applying Advanced Features
  - Feature Importance Evaluation
  - Hyperparameters Tuning
  - Feature Selection
- Many Iterations for Feature Engineering and Parameters Tuning

Predicting stock price is very interesting but difficult.
Notably, the classification is expected initially to be easier than regression.
However, a regressor is easy to follow the long-term trend of the prices, but the classifier is very hard to predict the price up/down in the daily small fluctuation.
A default [Ensemble Tree Gradient Boosting Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)[^GBR] with only the base features (`Open`, `High`, `Low`, `Close` and `Volume`) can easily follow the prices, but the predicted prices cannot provide good up/down predictions.
Predicting the nearer prices, e.g., `Open_next` (`Open` prices of the next day), is better.

[^GBR]: ["sklearn.ensemble.GradientBoostingRegressor," *scikit-learn.org*](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

<font size="1" style="line-height:11px;letter-spacing:0px">

    Close_next Regression r2-Score:  77.29%
    Close_next Regression to Up/Down Classification Accuracy Score:  55.71%
    Close_next Regression to Up/Down Classification       F1-Score:  71.56%

<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Up_predict</th>
      <th>Down_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Up_true</th>
      <td>39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Down_true</th>
      <td>31</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

                 precision    recall  f1-score   support
             Up       0.56      1.00      0.72        39
           Down       0.00      0.00      0.00        31
    avg / total       0.31      0.56      0.40        70
    
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Close_next</th>
      <th>Close_next_Regression (Test)</th>
      <th>Up_true</th>
      <th>Up_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-12</th>
      <td>2642.19</td>
      <td>2663.99</td>
      <td>2644.513344</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2018-04-13</th>
      <td>2663.99</td>
      <td>2656.30</td>
      <td>2666.069816</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

![png](fig/RgrCls.png)

    Open_next Regression r2-Score:  97.13%
    Open_next Regression to Up/Down Classification Accuracy Score:  78.57%
    Open_next Regression to Up/Down Classification       F1-Score:  80.52%
    
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Up_predict</th>
      <th>Down_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Up_true</th>
      <td>31</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Down_true</th>
      <td>7</td>
      <td>24</td>
    </tr>
  </tbody>
</table>

                 precision    recall  f1-score   support
             Up       0.82      0.79      0.81        39
           Down       0.75      0.77      0.76        31
    avg / total       0.79      0.79      0.79        70
    
<table border=0>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>Open_next</th>
      <th>Open_next_Regression (Test)</th>
      <th>Up_true</th>
      <th>Up_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-12</th>
      <td>2643.89</td>
      <td>2653.83</td>
      <td>2643.203038</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2018-04-13</th>
      <td>2653.83</td>
      <td>2676.90</td>
      <td>2664.924024</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</font>

![png](fig/RgrOpn.png)

### Improvement
The features importances shows that the model still pays too much attention to the previous days.
The models which consider the critical feature of time sequence, e.g., [Long Short-Term Memory (LSTM)](http://wikipedia.org/wiki/Long_short-term_memory)[^LSTM],
[Recurrent Neural Network (RNN)](http://wikipedia.org/wiki/Recurrent_neural_network)[^RNN], and the more advanced
[attention mechanism](http://arxiv.org/abs/1706.03762)[^att], should work better.
More important features of the global stock market, currency market, company status and financial related news, etc. and more data, maybe hourly prices, also should be considered.

[^LSTM]: ["Long Short-Term Memory (LSTM)," *Wikipedia*](http://wikipedia.org/wiki/Long_short-term_memory)

[^RNN]: ["Recurrent Neural Network (RNN)," *Wikipedia*](http://wikipedia.org/wiki/Recurrent_neural_network)

[^att]: [Vaswani et al. (Google), "Attention Is All You Need," *Conference on Neural Information Processing Systems (NIPS)*," 2017](http://arxiv.org/abs/1706.03762)

![png](fig/imp.png)