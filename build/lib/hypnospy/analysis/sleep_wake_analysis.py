from hypnospy import Wearable
from hypnospy import Experiment

import numpy as np


class SleepWakeAnalysis(object):

    def __init__(self, input: {Wearable, Experiment}):

        # TODO: we should have the option to only run sleep algorithms in the "hyp_night" period

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()


    @staticmethod
    def __get_activity(wearable, activityIdx):

        if wearable is None:
            raise ValueError("Invalid wearable!")

        if activityIdx is not None:

            if activityIdx in wearable.data.keys():
                return activityIdx
            else:
                raise ValueError("Activity %s not found in the dataset. "
                                 "Options are: %s" % (activityIdx, ','.join(wearable.data.keys())))

        else:
            act = wearable.get_activity_col()

        return act

    def run_sleep_algorithm(self, algname, activityIdx=None, rescoring=False, on_sleep_interval=False, inplace=False, **kargs):
        """ Function that chooses sleep algorithm from a list of heuristic (non-data driven) approaches
            Function takes in algorithm (1) choice, (2) resolution (30s or 60s) currently available
            and (3) activity index method. Currently counts and ENMO (universally derivable across triaxial
            devices is available).
            
            * Function indicates if webster rescoring rules are applied
            
        Parameters
        ----------
        algname: str
            Heuristic method of choice
            Available methods are: 'ScrippsClinic', 'Sadeh', 'Oakley10', 'ColeKripke','Sazonov' 
            Default is 'ScrippsClinic'.
        Returns
        -------
        
        Citations (methods used and implemented)
        --------
        [1]Kripke, D. F., Hahn, E. K., ... & Kline, L. E. (2010). Wrist actigraphic scoring for
        sleep laboratory patients: algorithm development. Journal of sleep research, 19(4), 612-619.
        [2]Sadeh, A., Sharkey, M. & Carskadon, M. A. Activity-based sleep-wake identification: an empirical
        test of methodological issues. Sleep 17, 201–207 (1994).
        [3]Tonetti L, Pasquini F, Fabbri M, Belluzzi M, Natale V (2008) Comparison 
        of two different actigraphs with polysomnography in healthy young subjects. Chronobiol Int 25: 145–153
        [4]Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J. & Gillin, J. C. Automatic sleep/wake 
        identification from wrist activity. Sleep 15, 461–469 (1992).
        [5]Sazonov, E. et al. Activity-based sleep-wake identification in infants. Physiol. Meas. 25, 1291 (2004).
        [6]Tilmanne, J., Urbain, J., Kothare, M. V., Wouwer, A. V., & Kothare, S. V. (2009). Algorithms for sleep–wake
        identification using actigraphy: a comparative study and new results. Journal of sleep research, 18(1), 85-98.
        
        """
        results = {}

        for wearable in self.wearables:

            activityIdx = SleepWakeAnalysis.__get_activity(wearable, activityIdx)
            activity = wearable.data[activityIdx]

            # Apply Scripps
            if algname.lower() == "scrippsclinic":
                scaler = 0.204
                if "scaler" in kargs:
                    scaler = kargs["scaler"]
                result = self.__scripps_clinic_algorithm(activity, scaler)

            # Apply Sadeh
            elif algname.lower() == "sadeh":
                params = {"min_threshold": 0, "minNat": 50, "maxNat": 100, "window_past": 6, "window_nat": 11,
                          "window_centered": 11}
                for p in params:
                    if p in kargs:
                        params[p] = kargs[p]
                result = self.__sadeh_algorithm(activity, **params)

            # Apply Oakley
            elif algname.lower() == "oakley":
                threshold = 10
                if "threshold" in kargs:
                    threshold = kargs["threshold"]
                result = self.__oakley_algorithm(activity, threshold=threshold)

            # Apply Cole-Kripke
            elif algname.lower() in ["colekripke", "cole-kripke"]:
                result = self.__cole_kripke_algorithm(activity)

            # Apply Sazonov
            elif algname.lower() in ["sazonov", "saznov"]:
                result = self.__sazonov_algorithm(activity)

            else:
                raise ValueError("Algorithm %s is unknown." % (algname))

            if rescoring:
                result = self.webster_rescoring_rules(result)

            if inplace:
                wearable.data[algname] = result

            results[wearable.get_pid()] = result

        return results

    # %%
    def run_all_sleep_algorithms(self, activityIdx=None, rescoring=False, inplace=False):
        """ This function runs the algorithm of choice"
        """

        results = {}

        for wearable in self.wearables:
            activityIdx = SleepWakeAnalysis.__get_activity(wearable, activityIdx)
            activity = wearable.data[activityIdx]

            result = dict()
            result["ScrippsClinic"] = self.__scripps_clinic_algorithm(activity)
            result["Sadeh"] = self.__sadeh_algorithm(activity)
            result["Oakley10"] = self.__oakley_algorithm(activity, threshold=10)
            result["ColeKripke"] = self.__cole_kripke_algorithm(activity)
            result["Sazonov"] = self.__sazonov_algorithm(activity)

            if rescoring:
                result["RescoredScrippsClinic"] = self.webster_rescoring_rules(result["ScrippsClinic"])
                result["RescoredSadeh"] = self.webster_rescoring_rules(result["Sadeh"])
                result["RescoredOakley10"] = self.webster_rescoring_rules(result["Oakley10"])
                result["RescoredColeKripke"] = self.webster_rescoring_rules(result["ColeKripke"])
                result["RescoredSazonov"] = self.webster_rescoring_rules(result["Sazonov"])

            if inplace:
                for col in result.keys():
                    wearable.data[col] = result[col]

            results[wearable.get_pid()] = result

        return results


    # Scripps Clinic Algorithm Definition
    def __scripps_clinic_algorithm(self, activity, scaler=0.204):

        act_series = dict()
        act_series["_a0"] = activity.fillna(0.0)

        # Enrich the dataframe with temporary values
        for i in range(1, 11):
            act_series["_a-%d" % (i)] = activity.shift(i).fillna(0.0)
            act_series["_a+%d" % (i)] = activity.shift(-i).fillna(0.0)

        # Calculates Scripps clinic algorithm
        scripps = scaler * (0.0064 * act_series["_a-10"] + 0.0074 * act_series["_a-9"] +
                            0.0112 * act_series["_a-8"] + 0.0112 * act_series["_a-7"] +
                            0.0118 * act_series["_a-6"] + 0.0118 * act_series["_a-5"] +
                            0.0128 * act_series["_a-4"] + 0.0188 * act_series["_a-3"] +
                            0.0280 * act_series["_a-2"] + 0.0664 * act_series["_a-1"] +
                            0.0300 * act_series["_a0"] + 0.0112 * act_series["_a+1"] +
                            0.0100 * act_series["_a+2"])

        scripps.name = "ScrippsClinic"
        # Returns a series with binary values: 1 for sleep, 0 for awake
        return (scripps < 1.0).astype(int)

    # %%
    # Sadeh Algorithm
    def __sadeh_algorithm(self, activity, min_threshold=0, minNat=50, maxNat=100, window_past=6, window_nat=11,
                          window_centered=11):
        """
        Sadeh model for classifying sleep vs active
        """

        act = activity.copy()

        _mean = act.rolling(window=window_centered, center=True, min_periods=1).mean()
        _std = act.rolling(window=window_past, min_periods=1).std()
        _nat = ((act >= minNat) & (act <= maxNat)).rolling(window=window_nat, center=True, min_periods=1).sum()

        _LocAct = (act + 1.).apply(np.log)

        sadeh = (7.601 - 0.065 * _mean - 0.056 * _std - 0.0703 * _LocAct - 1.08 * _nat)
        sadeh.name = "Sadeh"

        # Returns a series with binary values: 1 for sleep, 0 for awake
        return (sadeh > min_threshold).astype(int)

    # %%
    # Oakley Algorithm
    def __oakley_algorithm(self, activity=None, threshold=80):
        """
        Oakley method to class sleep vs active/awake
        """
        act = activity.copy()

        act_series = {}

        act_series["_a0"] = act.fillna(0.0)
        for i in range(1, 5):
            act_series["_a-%d" % (i)] = act.shift(i).fillna(0.0)
            act_series["_a+%d" % (i)] = act.shift(-i).fillna(0.0)

        oakley = 0.04 * act_series["_a-4"] + 0.04 * act_series["_a-3"] + 0.20 * act_series["_a-2"] + \
                 0.20 * act_series["_a-1"] + 2.0 * act_series["_a0"] + 0.20 * act_series["_a+1"] + \
                 0.20 * act_series["_a-2"] + 0.04 * act_series["_a-3"] + 0.04 * act_series["_a-4"]
        oakley.name = "Oakley%d" % (threshold)

        return (oakley <= threshold).astype(int)

    # %%
    # Cole Kripke algorithm
    def __cole_kripke_algorithm(self, activity):
        """
        Cole-Kripke method to classify sleep vs awake
        """
        act_series = {}

        act_series["_A0"] = activity.fillna(0.0)
        for i in range(1, 5):
            act_series["_A-%d" % (i)] = activity.shift(i).fillna(0.0)
        for i in range(1, 3):
            act_series["_A+%d" % (i)] = activity.shift(-i).fillna(0.0)

        w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [404, 598, 326, 441, 1408, 508, 350]
        p = 0.00001

        cole_kripke = p * (w_m4 * act_series["_A-4"] + w_m3 * act_series["_A-3"] +
                      w_m2 * act_series["_A-2"] + w_m1 * act_series["_A-1"] +
                      w_0 * act_series["_A0"] +
                      w_p1 * act_series["_A+1"] + w_p2 * act_series["_A+2"])

        cole_kripke.name = "Cole_Kripke"
        return (cole_kripke < 1.0).astype(int)

    # %%
    # Sazonov algorithm
    def __sazonov_algorithm(self, activity):
        """
        Sazonov formula as shown in the original paper
        """
        act_series = {}

        for w in range(1, 6):
            act_series["_w%d" % (w - 1)] = activity.rolling(window=w, min_periods=1).max()

        sazonov = 1.727 - 0.256 * act_series["_w0"] - 0.154 * act_series["_w1"] - \
                  0.136 * act_series["_w2"] - 0.140 * act_series["_w3"] - 0.176 * act_series["_w4"]
        sazonov.name = "Sazonov"
        return (sazonov >= 0.5).astype(int)


    # In the future include pre-trained ML/DL models here (optimize for features obtained)

    # Webster Rescoring Rules
    @staticmethod
    def webster_rescoring_rules(act, rescoring_rules="abcde"):

        if act.empty:
            return act

        haveAppliedAnyOtherRule = False

        if "a" in rescoring_rules or "A" in rescoring_rules:
            # After at least 4 minutes scored as wake, next minute scored as sleep is rescored wake
            # print "Processing rule A"
            maskA = act.shift(1).rolling(window=4, center=False,
                                         min_periods=1).sum() > 0  # avoid including actual period
            result = act.where(maskA, 0)
            haveAppliedAnyOtherRule = True

        if "b" in rescoring_rules or "B" in rescoring_rules:
            # After at least 10 minutes scored as wake, the next 3 minutes scored as sleep are rescored wake
            # print "Processing rule B"
            if haveAppliedAnyOtherRule == True:  # if this is true, I need to apply the next operation on the destination col
                act = result

            maskB = act.shift(1).rolling(window=10, center=False,
                                         min_periods=1).sum() > 0  # avoid including actual period
            result = act.where(maskB, 0).where(maskB.shift(1), 0).where(maskB.shift(2), 0)
            haveAppliedAnyOtherRule = True

        if "c" in rescoring_rules or "C" in rescoring_rules:
            # After at least 15 minutes scored as wake, the next 4 minutes scored as sleep are rescored as wake
            # print "Processing rule C"
            if haveAppliedAnyOtherRule == True:  # if this is true, I need to apply the next operation on the destination col
                act = result

            maskC = act.shift(1).rolling(window=15, center=False,
                                         min_periods=1).sum() > 0  # avoid including actual period
            result = act.where(maskC, 0).where(maskC.shift(1), 0).where(maskC.shift(2), 0).where(maskC.shift(3), 0)
            haveAppliedAnyOtherRule = True

        if "d" in rescoring_rules or "D" in rescoring_rules:
            # 6 minutes or less scored as sleep surroundeed by at least 10 minutes (before or after) scored as wake are rescored wake
            # print "Processing rule D"
            if haveAppliedAnyOtherRule == True:  # if this is true, I need to apply the next operation on the destination col
                act = result

            # First Part
            maskD1 = act.shift(1).rolling(window=10, center=False,
                                          min_periods=1).sum() > 0  # avoid including actual period
            tmpD1 = act.where(maskD1.shift(5), 0)
            haveAppliedAnyOtherRule = True

            # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
            maskD2 = act.shift(-10).rolling(window=10, center=False,
                                            min_periods=1).sum() > 0  # avoid including actual period
            tmpD2 = act.where(maskD2.shift(-5), 0)

            result = tmpD1 & tmpD2

        if "e" in rescoring_rules or "E" in rescoring_rules:
            # 10 minutes or less scored as sleep surrounded by at least 20 minutes (before or after) scored as wake are rescored wake
            # print "Processing rule E"
            if haveAppliedAnyOtherRule == True:  # if this is true, I need to apply the next operation on the destination col
                act = result

            # First Part
            maskE1 = act.shift(1).rolling(window=20, center=False,
                                          min_periods=1).sum() > 0  # avoid including actual period
            tmpE1 = act.where(maskE1.shift(9), 0)

            # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
            maskE2 = act.shift(-20).rolling(window=20, center=False,
                                            min_periods=1).sum() > 0  # avoid including actual period
            tmpE2 = act.where(maskE2.shift(-9), 0)

            result = tmpE1 & tmpE2

        return result
