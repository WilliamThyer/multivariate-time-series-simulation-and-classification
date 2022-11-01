import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd

from tsfresh.feature_extraction import extract_features  

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _make_time_series_a(time): 

    sin = np.sin(time)
    sin2 = np.sin(time*.02)
    mean_shift = time*.005
    signal_a1 = sin+sin2+mean_shift

    saw = sig.sawtooth(time)
    sin2 = np.sin(time*.02)
    mean_shift = time*.005
    signal_a2 = saw+sin2*-1+mean_shift*-1

    signal_a3 = sig.square(time*.5)*.2

    a = np.empty((3,len(time)))
    a[0] = signal_a1
    a[1] = signal_a2
    a[2] = signal_a3

    return a 

def _make_time_series_b(time):

    sin = np.sin(time*.05)
    sin2 = np.sin(time*.005)
    mean_shift = time*.005
    signal_b1 = sin+sin2+mean_shift

    saw = sig.sawtooth(time*.05)
    sin2 = np.sin(time*.002)
    mean_shift = time*.005
    signal_b2 = saw+sin2*-1+mean_shift*-1

    signal_b3 = sig.square(time*.5)*2

    b = np.empty((3,len(time)))
    b[0] = signal_b1
    b[1] = signal_b2
    b[2] = signal_b3

    return b 

def make_time_series_for_classification(time):

    a = _make_time_series_a(time)
    b = _make_time_series_b(time)

    return a, b

def add_noise(time_series, noise = .01):

    time_series[0] = time_series[0] + np.random.normal(0,scale=noise,size=len(time_series[0]))
    time_series[1] = time_series[1] + np.random.normal(0,scale=noise,size=len(time_series[1]))
    time_series[2] = time_series[2] + np.random.normal(0,scale=noise,size=len(time_series[2]))
        
    return time_series

def plot_signals(a, b):

    fig, axes = plt.subplots(6,1,figsize=(16,8),sharex=True)
    axes[0].plot(a[0])
    axes[1].plot(a[1])
    axes[2].plot(a[2])
    axes[3].plot(b[0],c='red')
    axes[4].plot(b[1],c='red')
    axes[5].plot(b[2],c='red')

    plt.show()

def convert_to_df(a, b):

    window_size = 20
    num_windows = int((a.shape[1]+b.shape[1])/window_size)
    windows = np.repeat(np.arange(num_windows),window_size)

    signal_a = pd.DataFrame(a.T)
    signal_a['label'] = 0 
    signal_b = pd.DataFrame(b.T)
    signal_b['label'] = 1

    signal_df = pd.concat([signal_a,signal_b]).reset_index(drop=True)
    signal_df['window'] = windows

    return signal_df

def extract_features_from_df(df):

    settings = {'mean':None,'standard_deviation':None,}
    df_extract = extract_features(
        df, 
        default_fc_parameters=settings,
        column_id='window',
        disable_progressbar=True)

    df_extract = df_extract.drop('label__standard_deviation',axis=1)
    df_extract = df_extract.rename({'label__mean':'label'},axis=1)

    return df_extract

def convert_to_xy(df):

    x = df.drop('label',axis=1).to_numpy()
    y = df['label'].to_numpy()

    shuffle = np.random.permutation(np.arange(len(y)))
    x = x[shuffle]
    y = y[shuffle]
    
    return x,y

def downsample(y,num_samples_per_class):

    idx = np.arange(len(y))
    idxa = idx[y==0][:num_samples_per_class]
    idxb = idx[y==1][:num_samples_per_class]

    idx_downsamp = np.concatenate([idxa,idxb])

    return x[idx_downsamp], y[idx_downsamp]
 
def make_prepared_dataset(time, noise):

    a,b = make_time_series_for_classification(time)
    a = add_noise(a,noise=noise)
    b = add_noise(b,noise=noise)

    df = convert_to_df(a,b)
    df_feat = extract_features_from_df(df)

    x,y = convert_to_xy(df_feat)

    return x, y

class DownsampleStratifiedShuffleSplit:

    def __init__(self, n_splits=3, test_size=.2, num_samples=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.shuffle = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=test_size)
        self.num_samples = num_samples

    def split(self, X, y, groups=None):

        for train_idx, test_idx in self.shuffle.split(X,y):

            downsamp0 = train_idx[y[train_idx]==0][:self.num_samples]
            downsamp1 = train_idx[y[train_idx]==1][:self.num_samples]
            train_idx_downsamp = np.concatenate([downsamp0, downsamp1])

            yield train_idx_downsamp, test_idx 

    def get_n_splits(self):
        return self.n_splits

