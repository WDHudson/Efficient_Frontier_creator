import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

stocks = ['NUS', 'HCSG', 'BEN', 'WST', 'SEIC', 'PB', 'FDS', 'APD', 'SPGI', 'JNJ', 'MSM', 'LANC', 'TROW', 'GGG', 'AOS']
# stocks = ['EGFIX', 'AKRIX', 'MBB', 'IEF', 'PSK', 'QQQ', 'RSP']

df = yf.download(stocks, start='2000-01-01', end='2021-06-29')

df = np.log(1+df['Adj Close'].pct_change())