import warnings

import numpy as np
import pandas as pd
import math

from hypnospy import Wearable, Experiment
from scipy.stats import entropy
from scipy import linalg  # linear algebra (matrix) processing package
from tqdm import tqdm, trange
from collections import defaultdict
#from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from CosinorPy import cosinor


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
        # TODO: this is extremely slow as it is now. At least we should try to do it with multithreads.
        print('=== Running SSA ===')
        ssas = []
        for idx, wearable in tqdm(enumerate(self.wearables)):
            if wearable.data.shape[0] == 0:
                warnings.warn("No data for PID %s. Skipping it." % wearable.get_pid())
                continue
            print(idx)
            ssas.append(self._get_SSA(wearable))
        return ssas

    # Extract SSA parameters, partial variances stored in ssa_io_pv, other stored in dict ssa, read from files in folder for
    # convenience, as full analysis takes a long time for each subject
    def _get_SSA(self, w, col='hyp_act_x', freq='15T'):
        ssa = defaultdict(dict)
        df = w.data[[w.get_experiment_day_col(), 'hyp_time_col', col]].resample('1T', on='hyp_time_col').mean()[col]

        ssa['r'], ssa['pv'], ssa['gk'], ssa['wm'] = self._get_SSA_par(df, L=1440)
        ssa['df'] = df

        # Joao says: not really sure why we should save these files:
        #pd.DataFrame(ssa[col]['gk'][:10, :]).to_csv('ssa_gk_' + self.filename + col + '.csv', index=False)
        #pd.DataFrame(ssa[col]['wm']).to_csv('ssa_wm_' + self.filename + col + '.csv', index=False)
        #pd.DataFrame(ssa[col]['pv']).to_csv('ssa_pv_' + self.filename + col + '.csv', index=False)

        df_1 = pd.DataFrame(np.transpose(ssa['gk']))

        # if self.data freq is different than df.resample, then the below line will cause an error
        # df_1 = df_1.set_index(self.data.index)
        df = w.data[[w.get_experiment_day_col(), 'hyp_time_col', col]].resample('1T', on='hyp_time_col').mean()[col]
        ssa['df'] = df

        # df_1 = df_1.set_index(df.index)
        df_1 = df_1.set_index(ssa['df'].index)

        df_2 = df_1.between_time('06:00', '23:00', include_start=True, include_end=True)
        df_gk = df_2[0] + df_2[1]

        # Get SSA acrophases
        ssa['acrophase'] = df_gk.resample('24H').agg(lambda x: np.nan if x.count() == 0 else x.idxmax())
        ssa['acrophase'].name = "acrophase"

        # Get vectors of coarse-grain params for ML
        ssa['gksum' + freq] = df_gk.resample(freq).mean()

        # Get trend i.e. mesor
        ssa['trend'] = df_1[0].resample('24H').mean()
        ssa['trend'].name = "trend"

        # Get period in minutes
        ssa['period'] = (ssa['acrophase'] - ssa['acrophase'].shift(1)).astype('timedelta64[m]')
        ssa["pid"] = w.get_pid()

        # Merge results
        results = pd.concat([ssa["acrophase"], ssa["trend"]], axis=1)

        return results, ssa

    @staticmethod
    def _get_SSA_par(df, L=1440, n_max_tries=3):  # 2 <= L <= N/2
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

        for t in range(n_max_tries):
            try:
                U, s, V = linalg.svd(X,
                                     full_matrices=True,
                                     compute_uv=True,
                                     overwrite_a=False,
                                     check_finite=True,
                                     lapack_driver='gesvd')
            except:
                continue

        if t == n_max_tries:
            raise ValueError("SSA reached the max number of tries with error.")

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

    def run_cosinor(self, col='hyp_act_x'):
        """
        returns a list with cosinor results
        """
        cosinors = []
        for idx, wearable in tqdm(enumerate(self.wearables)):

            if wearable.data.shape[0] == 0:
                warnings.warn("No data for PID %s. Skipping it." % wearable.get_pid())
                continue

            cosinors.append(self._get_cosinor(wearable, col))
        cosinors = pd.concat(cosinors)
        # Returns a dataframe indexed by ["pid", w.experiment_day]
        return cosinors.reset_index().set_index(["pid", cosinors.index.name])

    def _get_cosinor(self, w, col):
    
        freq = w.get_frequency_in_secs()
        s = w.data.groupby(w.get_experiment_day_col())['hyp_time_col'].transform(self._group_to_timepoints, freq)
        w.data['cosinor_timepoints'] = s

        cosinor_input = w.data[[w.get_experiment_day_col(), 'cosinor_timepoints', col]].copy()
        cosinor_input['cosinor_timepoints'] = s
        cosinor_input = cosinor_input[[w.get_experiment_day_col(), 'cosinor_timepoints', col]]
        cosinor_input.columns = [w.get_experiment_day_col(), 'x', 'y']

        cosinor_result = cosinor_input.groupby(w.get_experiment_day_col()).apply(self._apply_cosinor_on_exp_day)
        cosinor_result.index = cosinor_result.index.astype(int)

        # remove invalid days
        # cosinor_result = cosinor_result.loc[0:]

        cosinor_result["pid"] = w.get_pid()

        # cosinor_result is a tuple of
        # (pid, RegressionResultsWrapper, statistics, rhythm_params, X_test, Y_test, model)
        return cosinor_result

    @staticmethod
    def _apply_cosinor_on_exp_day(g, n_max_tries=3):
        for t in range(n_max_tries):
            try:
                cosinor_result = cosinor.fit_me(g['x'].values, g['y'].values, n_components=2, period=24,
                                                model_type='lin', lin_comp=True, alpha=0, plot=False)
            except:
                continue

        if t == n_max_tries:
            raise ValueError("SSA reached the max number of tries with error.")

        # cosinor_result[1] contains {p, p_reject, SNR, RSS, resid_SE, ME}
        # cosinor_result[2] contains {ME, period, amplitude, acrophase, mesor}
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
        pass

    def run_entropy(df, bins=20):
        bins = int(bins)
        hist, bin_edges = np.histogram(df, bins=bins)
        p = hist / float(hist.sum())
        ent = entropy(p)
        df['ent'] = ent
        return ent
