#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
from scipy.ndimage.morphology import binary_opening,binary_closing

def _create_inactivity_mask(data, duration, threshold):

    # Binary data
    binary_data = np.where(data >= threshold, 1, 0)

    # The first order diff Series indicates the indices of the transitions
    # between series of zeroes and series of ones.
    # Add zero at the beginning of this series to mark the beginning of the
    # first sequence found in the data.
    edges = np.concatenate([[0], np.diff(binary_data)])

    # Create the mask filled iwith ones by default.
    mask = np.ones_like(data)

    # Test if there is no edge (i.e. no consecutive zeroes).
    if all(e == 0 for e in edges):
        return pd.Series(mask, index=data.index)

    # Indices of upper transitions (zero to one).
    idx_plus_one = (edges > 0).nonzero()[0]
    # Indices of lower transitions (one to zero).
    idx_minus_one = (edges < 0).nonzero()[0]

    # Even number of transitions.
    if idx_plus_one.size == idx_minus_one.size:

        # Start with zeros
        if idx_plus_one[0] < idx_minus_one[0]:
            starts = np.concatenate([[0], idx_minus_one])
            ends = np.concatenate([idx_plus_one, [edges.size]])
        else:
            starts = idx_minus_one
            ends = idx_plus_one
    # Odd number of transitions
    # starting with an upper transition
    elif idx_plus_one.size > idx_minus_one.size:
        starts = np.concatenate([[0], idx_minus_one])
        ends = idx_plus_one
    # starting with an lower transition
    else:
        starts = idx_minus_one
        ends = np.concatenate([idx_plus_one, [edges.size]])

    # Index pairs (start,end) of the sequences of zeroes
    seq_idx = np.c_[starts, ends]
    # Length of the aforementioned sequences
    seq_len = ends - starts

    for i in seq_idx[np.where(seq_len >= duration)]:
        mask[i[0]:i[1]] = 0

    return pd.Series(mask, index=data.index)

def _padded_data(data, value, periods, frequency):

    date_offset = pd.DateOffset(seconds=frequency.total_seconds())
    pad_beginning = pd.Series(
        data=value,
        index=pd.date_range(
            end=data.index[0],
            periods=periods,
            freq=date_offset,
            closed='left'
        ),
        dtype=float
    )
    pad_end = pd.Series(
        data=value,
        index=pd.date_range(
            start=data.index[-1],
            periods=periods,
            freq=date_offset,
            closed='right'
        ),
        dtype=float
    )
    return pd.concat([pad_beginning, data, pad_end])

def Crespo(self, zeta=20, zeta_r=40, zeta_a=10,t=.33, alpha='7h', beta='1h',
        estimate_zeta=False, seq_length_max=100, verbose=False):

    r"""Crespo algorithm for activity/rest identification
    Algorithm for automatic identification of activity-rest periods based
    on actigraphy, developped by Crespo et al. [1]_.
    Parameters
    ----------
    zeta: int, optional
    Maximum number of consecutive zeroes considered valid.
    Default is 15.
    zeta_r: int, optional
    Maximum number of consecutive zeroes considered valid (rest).
    Default is 30.
    zeta_a: int, optional
    Maximum number of consecutive zeroes considered valid (active).
    Default is 2.
    t: float, optional
    Percentile for invalid zeroes.
    Default is 0.33.
    alpha: str, optional
    Average hours of sleep per night.
    Default is '8h'.
    beta: str, optional
    Length of the padding sequence used during the processing.
    Default is '1h'.
    estimate_zeta: bool, optional
    If set to True, zeta values are estimated from the distribution of
    ratios of the number of series of consecutive zeroes to
    the number of series randomly chosen from the actigraphy data.
    Default is False.
    seq_length_max: int, optional
    Maximal length of the aforementioned random series.
    Default is 100.
    verbose: bool, optional
    If set to True, print the estimated values of zeta.
    Default is False.
    Returns
    -------
    crespo : pandas.core.Series
    Time series containing the estimated periods of rest (0) and
    activity (1).
    References
    ----------
    .. [1] Crespo, C., Aboy, M., Fernández, J. R., & Mojón, A. (2012).
    Automatic identification of activity–rest periods based on
    actigraphy. Medical & Biological Engineering & Computing, 50(4),
    329–340. http://doi.org/10.1007/s11517-012-0875-y

    """

    # 1. Pre-processing
    # This stage produces an initial estimate of the rest-activity periods

    # 1.1. Signal conditioning based on empirical probability model
    # This step replaces sequences of more than $\zeta$ "zeroes"
    # with the t-percentile value of the actigraphy data
    # zeta = 15
    if estimate_zeta:
        zeta = _estimate_zeta(self.data['ENMO'], seq_length_max)
        if verbose:
            print("CRESPO: estimated zeta = {}".format(zeta))
    # Determine the sequences of consecutive zeroes
    mask_zeta = _create_inactivity_mask(self.data['ENMO'], zeta, 1)

    # Determine the user-specified t-percentile value
    s_t = self.data['ENMO'].quantile(t)

    # Replace zeroes with the t-percentile value
    x = self.data['ENMO'].copy()
    x[mask_zeta > 0] = s_t

    # Median filter window length L_w
    frequency = pd.Timedelta(np.diff(self.data.index.values).min())
    
    L_w = int(pd.Timedelta(alpha)/frequency)+1
    #print(L_w)
    #L_w = int(pd.Timedelta(alpha)/self.frequency)+1
    L_w_over_2 = int((L_w-1)/2)
    #print(L_w_over_2)

    # Pad the signal at the beginning and at the end with a sequence of
    # $\alpha/2$ h of elements of value $m = max(s(t))$.
    #
    # alpha_epochs = int(pd.Timedelta(alpha)/self.frequency)
    # alpha_epochs_half = int(alpha_epochs/2)
    # beta_epochs = int(pd.Timedelta(beta)/self.frequency)

    s_t_max = self.data['ENMO'].max()
    #print(s_t_max)
    x_p = _padded_data(x, s_t_max, L_w_over_2, frequency)
    #print(len(x_p))

    # 1.2 Rank-order processing and decision logic
    # Apply a median filter to the $x_p$ series
    x_f = x_p.rolling(L_w, center=True).median()

    # Rank-order thresholding
    # Create a series $y_1(n)$ where $y_1(n) = 1$ for $x_f(n)>p$, $0$ otw.
    # The threshold $p$ is the percentile of $x_f(n)$ corresponding to
    # $(h_s/24)\times 100\%$
    p_threshold = x_f.quantile((pd.Timedelta(alpha)/pd.Timedelta('24h')))
    y_1 = pd.Series(np.where(x_f > p_threshold, 1, 0), index=x_f.index)
    # 1.3 Morphological filtering
    # Morph. filter window length, L_p
    L_p = int(pd.Timedelta(beta)/frequency)+1
    # Morph. filter, M_f
    M_f = np.ones(L_p)
    # Apply a morphological closing operation
    y_1_close = binary_closing(y_1, M_f).astype(int)
    # Apply a morphological opening operation
    y_1_close_and_open = binary_opening(y_1_close, M_f).astype(int)
    y_e = pd.Series(y_1_close_and_open, index=y_1.index)
    #print(self.data['ENMO'])
    #print(y_e)
    # 2. Processing and decision logic
    # This stage uses the estimates of the rest-activity periods
    # from the previous stage.
    # 2.1 Model-based data validation
    # Create a mask for sequences of more than $\zeta_{rest}$ zeros
    # during the rest periods
    # zeta_r = 30
    # zeta_a = 2
    if estimate_zeta:
        zeta_r = _estimate_zeta(self.data['ENMO'][y_e < 1], seq_length_max)
        zeta_a = _estimate_zeta(self.data['ENMO'][y_e > 0], seq_length_max)
        if verbose:
            print("CRESPO: estimated zeta@rest= {}".format(zeta_r))
            print("CRESPO: estimated zeta@actv= {}".format(zeta_a))

    # Find sequences of zeroes during the rest and the active periods
        # and mark as invalid sequences of more $\zeta_x$ zeroes.

        # Use y_e series as a filter for the rest periods
            
    cop = self.data['ENMO'].copy()
    mask_rest = _create_inactivity_mask(cop[y_e < 1], zeta_r, 1 )

        # Use y_e series as a filter for the active periods
    mask_actv = _create_inactivity_mask(cop[y_e > 0], zeta_a, 1)

    mask = pd.concat([mask_actv, mask_rest], verify_integrity=True)

        # 2.2 Adaptative rank-order processing and decision logic

        # Replace masked values by NaN so that they are not taken into account
        # by the median filter.
        # Done before padding to avoid unaligned time series.

    x_nan = self.data['ENMO'].copy()
    x_nan[mask < 1] = np.NaN

        # Pad the signal at the beginning and at the end with a sequence of 1h
        # of elements of value m = max(s(t)).
    x_sp = _padded_data(x_nan, s_t_max,L_p-1,frequency)
        
        # Apply an adaptative median filter to the $x_{sp}$ series

        # no need to use a time-aware window as there is no time gap
        # in this time series by definition.
    x_fa = x_sp.rolling(L_w, center=True, min_periods=L_p-1).median()

        # The 'alpha' hour window is biased at the edges as it is not
        # symmetrical anymore. In the regions (start, start+alpha/2,
        # the median needs to be calculate by hand.
        # The range is start, start+alpha as the window is centered.
    median_start = x_sp.iloc[0:L_w].expanding(center=True).median()
    median_end = x_sp.iloc[-L_w-1:-1].sort_index(ascending=False).expanding(center=True).median()[::-1]

        # replace values in the original x_fa series with the new values
        # within the range (start, start+alpha/2) only.
    x_fa.iloc[0:L_w_over_2] = median_start.iloc[0:L_w_over_2]
    x_fa.iloc[-L_w_over_2-1:-1] = median_end.iloc[0:L_w_over_2]

        # restore original time range
    x_fa = x_fa[self.data.index[0]:self.data.index[-1]]

    p_threshold = x_fa.quantile((pd.Timedelta(alpha)/pd.Timedelta('24h')))

    y_2 = pd.Series(np.where(x_fa > p_threshold, 1, 0), index=x_fa.index)

        # ### 2.3 Morphological filtering
    y_2_close = binary_closing(y_2,structure=np.ones(2*(L_p-1)+1)).astype(int)

    y_2_open = binary_opening(y_2_close,structure=np.ones(2*(L_p-1)+1)).astype(int)

    crespo = pd.Series(y_2_open,index=y_2.index)

        # Manual post-processing
    crespo.iloc[0] = 1
    crespo.iloc[-1] = 1
    self.data['crespo'] = crespo
    
    diff = crespo.diff(1)
    crespo_on = crespo[diff == 1].index
    crespo_off = crespo[diff == -1].index
    #Temporary fix for detecting rest at the start and end of the data, just deletes these intervals
    crespo_on = crespo_on[1:-1]
    crespo_off = crespo_off[1:-1]
    self.crespo_on = crespo_on
    self.crespo_off = crespo_off
        
    return self