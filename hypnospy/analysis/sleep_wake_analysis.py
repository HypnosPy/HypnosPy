from hypnospy import Wearable

class SleepWakeAnalysis(object):

    def __init__(self, wearable: Wearable):

        self.wearable = wearable

        # Get activity col configuration from wearable
        # TODO: Assuming we are using only the first axis
        self.activityIdx = wearable.activitycols[0]
        print("Using col %s as activity index." % (self.activityIdx))

        if not self.wearable.is_act_count:
            raise AttributeError("This device does not have activity counts. Sleep Formulas rely on activity counts.")

        # TODO: we should have the option to only run sleep algorithms in the "hyp_night" period


    def run_sleep_algorithm(self, algname, activityIdx=None, resolution="30s", rescoring=False):
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

        #Apply Scripps
        if algname == "ScrippsClinic":
            result = self.scripps_clinic_algorithm(activityIdx)

        #Apply Sadeh
        elif algname == "Sadeh":
            result = self.sadeh_algorithm(activityIdx)

        # Apply Oakley
        elif algname == "Oakley10":
            result = self.oakley_algorithm(activityIdx, threshold=10)

        # Apply Cole-Kripke
        elif algname == "ColeKripke":
            result = self.cole_kripke_algorithm(activityIdx)

        # Apply Sazonov
        elif algname == "Sazonov":
            result = self.sazonov_algorithm(activityIdx)

        else:
            print("ALGORITHM %s NOT IMPLEMENTED." % (algname))

        if rescoring:
            result = self.webster_rescoring_rules(result)

        return result

    # %%
    def run_sleep_all_algorithm(self, activityIdx=None, rescoring=False):
        """ This function runs the algorithm of choice"
        """

        result = dict()
        result["ScrippsClinic"] = self.scripps_clinic_algorithm(activityIdx)
        result["Sadeh"] = self.sadeh_algorithm(activityIdx)
        result["Oakley10"] = self.oakley_algorithm(activityIdx, threshold=10)
        result["ColeKripke"] = self.cole_kripke_algorithm(activityIdx)
        result["Sazonov"] = self.sazonov_algorithm(activityIdx)

        if rescoring:
            result["RescoredScrippsClinic"] = self.webster_rescoring_rules(result["ScrippsClinic"])
            result["RescoredSadeh"] = self.webster_rescoring_rules(result["Sadeh"])
            result["RescoredOakley10"] = self.webster_rescoring_rules(result["Oakley10"])
            result["RescoredColeKripke"] = self.webster_rescoring_rules(result["ColeKripke"])
            result["RescoredSazonov"] = self.webster_rescoring_rules(result["Sazonov"])

        return result

    ### INCLUDE OUR HR- RESCORING HERE?####


    def __get_activity(self, activityIdx):
        if activityIdx is None:
            act = self.wearable.data[self.activityIdx]
        else:
            if activityIdx in self.wearable.data.keys():
                act = self.wearable.data[activityIdx]
            else:
                raise ValueError("Activity %s not found in the dataset. "
                                 "Options are: %s" % (activityIdx, ','.join(self.wearable.data.keys())))

        return act


    #Scripps Clinic Algorithm Definition
    def scripps_clinic_algorithm(self, activityIdx=None, scaler=0.204):

        act = self.__get_activity(activityIdx)

        act_series = dict()
        act_series["_a0"] = act.fillna(0.0)

        # Enrich the dataframe with temporary values
        for i in range(1,11):
            act_series["_a-%d" % (i)] = act.shift(i).fillna(0.0)
            act_series["_a+%d" % (i)] = act.shift(-i).fillna(0.0)

        # Calculates Scripps clinic algorithm
        scripps = scaler * (0.0064 * act_series["_a-10"] + 0.0074 * act_series["_a-9"] +
                        0.0112 * act_series["_a-8"] + 0.0112 * act_series["_a-7"] +
                        0.0118 * act_series["_a-6"] + 0.0118 * act_series["_a-5"] +
                        0.0128 * act_series["_a-4"] + 0.0188 * act_series["_a-3"] +
                        0.0280 * act_series["_a-2"] + 0.0664 * act_series["_a-1"] +
                        0.0300 * act_series["_a0"] + 0.0112 * act_series["_a+1"] +
                        0.0100 * act_series["_a+2"])

        # Returns a series with binary values: 1 for sleep, 0 for awake
        return (scripps < 1.0).astype(int)


    # %%
    # Sadeh Algorithm
    def sadeh_algorithm(self, activityIdx=None, min_threshold=0, minNat=50, maxNat=100,
                        window_past = 6, window_nat = 11, window_centered = 11):
        """
        Sadeh model for classifying sleep vs active
        """
        act = self.__get_activity(activityIdx)

        _mean = act.rolling(window=window_centered, center=True, min_periods=1).mean()
        _std = act.rolling(window=window_past, min_periods=1).std()
        _nat = ((act >= minNat) & (act <= maxNat)).rolling(window=window_nat, center=True, min_periods=1).sum()

        _LocAct = (act + 1.).apply(np.log)

        sadeh = (7.601 - 0.065 * _mean - 0.056 * _std - 0.0703 * _LocAct - 1.08 * _nat)

        # Returns a series with binary values: 1 for sleep, 0 for awake
        return (sadeh > min_threshold).astype(int)


    # %%
    # Oakley Algorithm
    def oakley_algorithm(self, activityIdx=None, threshold=80):
        """
        Oakley method to class sleep vs active/awake
        """
        act = self.__get_activity(activityIdx)

        act_series = {}

        act_series[activityIdx] = act.fillna(0.0)
        for i in range(1,5):
            act_series["_a-%d" % (i)] = act.shift(i).fillna(0.0)
            act_series["_a+%d" % (i)] = act.shift(-i).fillna(0.0)

        oakley = 0.04 * act_series["_a-4"] + 0.04 * act_series["_a-3"] + 0.20 * act_series["_a-2"] + \
                 0.20 * act_series["_a-1"] + 2.0 * act_series[activityIdx] + 0.20 * act_series["_a+1"] + \
                 0.20 * act_series["_a-2"] + 0.04 * act_series["_a-3"] + 0.04 * act_series["_a-4"]

        return (oakley <= threshold).astype(int)


    # %%
    # Cole Kripke algorithm
    def cole_kripke_algorithm(self, activityIdx=None):
        """
        Cole-Kripke method to classify sleep vs awake
        """
        act = self.__get_activity(activityIdx)

        act_series = {}

        act_series["_A0"] = act.fillna(0.0)
        for i in range(1,5):
            act_series["_A-%d" % (i)] = act.shift(i).fillna(0.0)
        for i in range(1,3):
            act_series["_A+%d" % (i)] = act.shift(-i).fillna(0.0)

        w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [404, 598, 326, 441, 1408, 508, 350]
        p = 0.00001

        result = p * (w_m4 * act_series["_A-4"] + w_m3 * act_series["_A-3"] +
                      w_m2 * act_series["_A-2"] + w_m1 * act_series["_A-1"] +
                      w_0 * act_series["_A0"] +
                      w_p1 * act_series["_A+1"] + w_p2 * act_series["_A+2"])

        return (result < 1.0).astype(int)

    # %%
    # Sazonov algorithm
    def sazonov_algorithm(self, activityIdx=None):
        """
        Sazonov formula as shown in the original paper
        """
        act = self.__get_activity(activityIdx)

        act_series = {}

        for w in range(1,6):
            act_series["_w%d" % (w-1)] = act.rolling(window=w, min_periods=1).max()

        sazonov = 1.727 - 0.256 * act_series["_w0"] - 0.154 * act_series["_w1"] -\
                  0.136 * act_series["_w2"] - 0.140 * act_series["_w3"] - 0.176 * act_series["_w4"]

        return (sazonov >= 0.5).astype(int)


    # %% THIS NEEDS SOME TWEEKING, WE SHOULD ALSO ADD AN ENSEMBLE THAT IS ON ABSOLUTE AGREEMENT
    # Ensemble model for sleep algorithms
    def sleep_likelihood_by_ensemble(self, algorithms):
        """
        From a list of algorithm, calculate the lihelihood of sleep (i.e, % votes)
        """
        # TODO: need to be tested
        return algorithms.sum(axis=1) / len(algorithms)

    # In the future include pre-trained ML/DL models here (optimize for features obtained)

    # Webster Rescoring Rules
    def webster_rescoring_rules(act, rescoring_rules="abcde"):

        if act.empty:
            return act

        haveAppliedAnyOtherRule = False

        if "a" in rescoring_rules or "A" in rescoring_rules:
            # After at least 4 minutes scored as wake, next minute scored as sleep is rescored wake
            #print "Processing rule A"
            maskA = act.shift(1).rolling(window=4, center=False, min_periods=1).sum() > 0 # avoid including actual period
            result = act.where(maskA, 0)
            haveAppliedAnyOtherRule = True

        if "b" in rescoring_rules or "B" in rescoring_rules:
            # After at least 10 minutes scored as wake, the next 3 minutes scored as sleep are rescored wake
            #print "Processing rule B"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                act = result

            maskB = act.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
            result = act.where(maskB, 0).where(maskB.shift(1), 0).where(maskB.shift(2), 0)
            haveAppliedAnyOtherRule = True

        if "c" in rescoring_rules or "C" in rescoring_rules:
            # After at least 15 minutes scored as wake, the next 4 minutes scored as sleep are rescored as wake
            #print "Processing rule C"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                act = result

            maskC = act.shift(1).rolling(window=15, center=False, min_periods=1).sum() > 0 # avoid including actual period
            result = act.where(maskC, 0).where(maskC.shift(1), 0).where(maskC.shift(2), 0).where(maskC.shift(3), 0)
            haveAppliedAnyOtherRule = True

        if "d" in rescoring_rules or "D" in rescoring_rules:
            # 6 minutes or less scored as sleep surroundeed by at least 10 minutes (before or after) scored as wake are rescored wake
            #print "Processing rule D"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                act = result

            # First Part
            maskD1 = act.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpD1 = act.where(maskD1.shift(5), 0)
            haveAppliedAnyOtherRule = True

            # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
            maskD2 = act.shift(-10).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpD2 = act.where(maskD2.shift(-5), 0)

            result = tmpD1 & tmpD2

        if "e" in rescoring_rules or "E" in rescoring_rules:
            # 10 minutes or less scored as sleep surrounded by at least 20 minutes (before or after) scored as wake are rescored wake
            #print "Processing rule E"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                act = result

            # First Part
            maskE1 = act.shift(1).rolling(window=20, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpE1 = act.where(maskE1.shift(9), 0)

            # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
            maskE2 = act.shift(-20).rolling(window=20, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpE2 = act.where(maskE2.shift(-9), 0)

            result = tmpE1 & tmpE2

        return result
