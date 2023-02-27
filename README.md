# Synthetic financial time series generation with regime clustering
### Authors: Kirill Zakharov, Elizaveta Stavinova
Official realisation by our article (link will be later).

## Installation
To use the method, you must install following Python libraries:
```
pip install ruptures
pip install stumpy
pip install scipy
pip install sklearn
pip install torch torchvision
pip install yfinance
pip install pyts
pip install statsmodels
pip install fbprophet
```

## Method
General pipeline of our method is presented below. The main idea is to use the clustering approach on allocated regimes. For a detail description see the article.
![Pipeline](https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/pipeline_V3-1.png)

We also proposed the modification of existing GAN architectures, adding Supervisor and second Discriminator.
<p align="center">
<img src="https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/CLS-GAN_Pipeline-1.png"  width="60%" height="30%">
</p>

## Experiments
For the experiments we have used three open access datasets which describes stock prices. All data available in folder *Data* and even more [here](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs).
For the quality assessment we have used distribution statistics (skewness, kurtosis), sum of absolute squared values of spectral density, Jensen-Shannon divergence, two-sample Kolmogorov-Smirnov test statistic, local extrema, autocorrelation and machine learning metrics (MSE on forecasting by time series cross-validation).

<p align="center">
<img src="https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/table.png"  width="60%" height="30%">
</p>

It can be seen from the autocorrelation plots that our approach gives a better approximation during the lags.
<p align="center">
<img src="https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/autocorr.png"  width="60%" height="30%">
</p>

Obtained Q–Q plots of the extremum points in synthetic and the corresponding initial time series:
<p align="center">
<img src="https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/local_extrema.png"  width="70%">
</p>

On the figure below presented distributions: the original one and the distribution of revenues in daily (differentiated time series) and monthly (differentiated with time lag of 20 days time series) scales.
<p align="center">
<img src="https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/distributions.png"  width="60%" height="30%">
</p>

## Hyperparameters
For training procedure you can use the following hyperparameters.
<p align="center">
<img src="https://github.com/AlgoMathITMO/CLSGAN/blob/main/images/hypers.png"  width="60%" height="30%">
</p>
