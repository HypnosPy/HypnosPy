import numpy as np
import pandas as pd
import math

from hypnospy import Wearable, Experiment
from scipy.stats import entropy
from scipy import linalg  # linear algebra (matrix) processing package
from tqdm import tqdm, trange
from collections import defaultdict
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from CosinorPy import file_parser, cosinor, cosinor1


class CircadianAnalysis(object):

    def __init__(self, input: {Wearable, Experiment}):
        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

    def run_SSA(self):
        """
        sets ssa results in w.ssa
        """
        print('=== Running SSA ===')
        for idx, w in tqdm(enumerate(self.wearables)):
            # print(wearable.get_pid())

            if w.data.shape[0] == 0:
                warnings.warn("No data for PID %s. Skipping it." % wearable.get_pid())
                continue

            print(idx)
            w = self._get_SSA(w)
            self.wearables[idx] = w

    # Extract SSA parameters, partial variances stored in ssa_io_pv, other stored in dict ssa, read from files in folder for
    # convenience, as full analysis takes a long time for each subject
    def _get_SSA(self, w, cols=['hyp_act_x'], freqs=['15T']):
        ssa = defaultdict(dict)
        for col in cols:
            df = w.data[[w.get_experiment_day_col(), 'hyp_time_col', 'hyp_act_x']].resample('1T', on='hyp_time_col').mean()
            df = df['hyp_act_x']
            ssa[col]['r'], ssa[col]['pv'], ssa[col]['gk'], ssa[col]['wm'] = self._get_SSA_par(df, L=1440)
            ssa[col]['df'] = df

            pd.DataFrame(ssa[col]['gk'][:10, :]).to_csv('ssa_gk_' + 'self.filename' + col + '.csv', index=False)
            pd.DataFrame(ssa[col]['wm']).to_csv('ssa_wm_' + 'self.filename' + col + '.csv', index=False)
            pd.DataFrame(ssa[col]['pv']).to_csv('ssa_pv_' + 'self.filename' + col + '.csv', index=False)
            for freq in freqs:
                df_1 = pd.DataFrame(np.transpose(ssa[col]['gk']))

                # if self.data freq is different than df.resample, then the below line will cause an error
                # df_1 = df_1.set_index(self.data.index)
                df = w.data[[w.get_experiment_day_col(), 'hyp_time_col', 'hyp_act_x']].resample('1T', on='hyp_time_col').mean()
                df = df['hyp_act_x']
                ssa[col]['df'] = df

                # df_1 = df_1.set_index(df.index)
                df_1 = df_1.set_index(ssa[col]['df'].index)

                df_2 = df_1.between_time('06:00', '23:00', include_start=True, include_end=True)
                df_gk = df_2[0] + df_2[1]
                # Get SSA acrophases
                ssa[col]['acrophase'] = df_gk.resample('24H').agg(lambda x: np.nan if x.count() == 0 else x.idxmax())
                # Get vectors of coarse-grain params for ML
                ssa[col]['gksum' + freq] = df_gk.resample(freq).mean()
                # Get trend i.e. mesor
                ssa[col]['trend'] = df_1[0].resample('24H').mean()
                # Get period in minutes
                ssa[col]['period'] = (ssa[col]['acrophase'] - ssa[col]['acrophase'].shift(1)).astype('timedelta64[m]')
        w.ssa = ssa
        return w

    @staticmethod
    def _get_SSA_par(df, L=1440):  # 2 <= L <= N/2
        N = len(df)
        K = N - L + 1

        dataset = timeseries_dataset_from_array(
            data=df,
            targets=None,
            sequence_length=L,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=len(df)
        )

        X = list(dataset.as_numpy_iterator())[0]
        print(X.shape)

        try:
            U, s, V = linalg.svd(X,
                                 full_matrices=True,
                                 compute_uv=True,
                                 overwrite_a=False,
                                 check_finite=True,
                                 lapack_driver='gesvd')
        except:
            U, s, V = linalg.svd(X,
                                 full_matrices=True,
                                 compute_uv=True,
                                 overwrite_a=False,
                                 check_finite=True,
                                 lapack_driver='gesvd')

        l = s ** 2  # partial variances
        r = len(s)  # np.linalg.matrix_rank(X) # matrix rank and total number of components
        ### time-series components ###
        gkList = np.zeros(shape=(r, N))  # zero matrix in whose rows SSA components will be saved

        print('input:', X.shape)
        print('U:', U.shape)
        print('s:', s.shape)
        print('V:', V.shape)

        print('r:', r)
        print('gkList:', gkList.shape)

        for k in trange(r, position=0, leave=True):
            Uk = U[:, k]  # k-th order column singular vector
            Vk = V[k, :]  # k-th order row singular vector
            Xk = s[k] * np.outer(Uk, Vk)  # k-th order matrix component
            gk = []  # empty array in which to save successive k-th order component values
            for i in range(min(K - 1, L - 1), -max(K - 1, L - 1) - 1, -1):  # loop over diagonals
                gki = np.mean(np.diag(np.fliplr(Xk), i))  # successive time.series values
                gk.append(gki)
            gkList[k] = gk  # k-th order component

        ### w-corr matrix ###
        w = []  # empty array to which to add successive weights
        LL = min(L, K)
        KK = max(L, K)
        for ll in range(1, LL + 1):  # first 1/3 part of weights
            w.append(ll)
        for ll in range(LL + 1, KK + 1):  # second 1/3 part of weights
            w.append(LL)
        for ll in range(KK + 1, N + 1):  # third 1/3 part of weights
            w.append(N - ll)
        kMin = kkMin = 0  # show w-corr matrix for first 20 index values
        kMax = kkMax = 20

        wMatrix = [[sum(w * gkList[k] * gkList[kk]) / (
                    math.sqrt(sum(w * gkList[k] * gkList[k])) * math.sqrt(sum(w * gkList[kk] * gkList[kk]))) for k in
                    range(kMin, kMax)] for kk in range(kkMin, kkMax)]
        wMatrix = np.array(wMatrix)
        return (r, l, gkList, wMatrix);

    def run_cosinor(self):
        """
        sets cosinor results in w.cosinor
        """

        for idx, w in tqdm(enumerate(self.wearables)):

            if w.data.shape[0] == 0:
                warnings.warn("No data for PID %s. Skipping it." % wearable.get_pid())
                continue

            print(idx)
            w = self._get_cosinor(w)
            self.wearables[idx] = w

    def _get_cosinor(self, w, col='hyp_act_x', freqs=2):
    
        freq = w.get_frequency_in_secs()
        s = w.data.groupby(w.get_experiment_day_col())['hyp_time_col'].transform(self._group_to_timepoints, freq)
        w.data['cosinor_timepoints'] = s

        cosinor_input = w.data[[w.get_experiment_day_col(), 'cosinor_timepoints', col]].copy()
        cosinor_input['cosinor_timepoints'] = s
        cosinor_input = cosinor_input[[w.get_experiment_day_col(), 'cosinor_timepoints', col]]
        cosinor_input.columns = ['test', 'x', 'y']
        cosinor_input['test'] = cosinor_input['test'].astype(str)

        cosinor_result = cosinor_input.groupby('test').apply(self._apply_cosinor_on_exp_day)

        # cosinor_result is a tuple of
        # (RegressionResultsWrapper, statistics, rhythm_params, X_test, Y_test, model) 
        w.cosinor = cosinor_result
        return w

    @staticmethod
    def _apply_cosinor_on_exp_day(g):
        try:
            cosinor_result = cosinor.fit_me(g['x'].values,
                       g['y'].values, 
                       n_components=2, 
                       period=24, 
                       model_type='lin', 
                       lin_comp=True, 
                       alpha=0,
                       plot=False)
        except:
            cosinor_result = cosinor.fit_me(g['x'].values,
                       g['y'].values, 
                       n_components=2, 
                       period=24, 
                       model_type='lin', 
                       lin_comp=True, 
                       alpha=0,
                       plot=False)
        # cosinor_result[1] contains {p, p_reject, SNR, RSS, resid_SE, ME}
        # cosinor_result[2] contains {ME, period, aplitude, acrophase, mesor}
        # Note: sometimes, acrophase is NaN.
        df = pd.concat([pd.Series(cosinor_result[1]), pd.Series(cosinor_result[2])], axis=0)
        return df

    @staticmethod
    def _group_to_timepoints(g, freq=2):
        hyp_exp_day_length = g.shape[0]
        timepoints = np.arange(0, hyp_exp_day_length*freq, freq)
        return timepoints

    def run_PSD():
        # apply power spectral density analysis
        df['PSD'] = PSD
        pass

    def run_entropy(df, bins=20):
        bins = int(bins)
        hist, bin_edges = np.histogram(df, bins=bins)
        p = hist / float(hist.sum())
        ent = entropy(p)
        df['ent'] = ent
        return ent
