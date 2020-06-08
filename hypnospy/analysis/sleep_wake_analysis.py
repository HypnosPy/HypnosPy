from hypnospy import Wearable

class SleepWakeAnalysis(object):

    def __init__(self, wearable: Wearable):
        self.wearable = wearable


    def apply_sleep_algorithm(df, algname, activity_col="ENMO", resolution="30s" ,rescoring=False):
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
            df["ScrippsClinic"] = scripps_clinic_algorithm(df, activity_col)
            if rescoring:
                df["RescoredScrippsClinic"] = webster_rescoring_rules(df["ScrippsClinic"])

        #Apply Sadeh
        elif algname == "Sadeh":
            df["Sadeh"] = sadeh_algorithm(df, activity_col)
            if rescoring:
                df["RescoredSadeh"] = webster_rescoring_rules(df["Sadeh"])

        # Apply Oakley
        elif algname == "Oakley10":
            df["Oakley10"] = oakley_algorithm(df, activity_col, threshold=10)
            if rescoring:
                df["RescoredOakley10"] = webster_rescoring_rules(df["Oakley10"])

        # Apply Cole-Kripke
        elif algname == "ColeKripke":
            df["ColeKripke"] = cole_kripke_algorithm(df, activity_col)
            if rescoring:
                df["RescoredColeKripke"] = webster_rescoring_rules(df["ColeKripke"])

        # Apply Sazonov
        elif algname == "Sazonov":
            df["Sazonov"] = sazonov_algorithm(df, activity_col)
            if rescoring:
                df["RescoredSazonov"] = webster_rescoring_rules(df["Sazonov"])

        else:
            print("ALGORITHM %s NOT IMPLEMENTED." % (algname))


    # %%
    def run_sleepalgorithms(df, activity_col="ENMO"):
        """ This function runs the algorithm of choice"
        """
        #Apply Scripps    
        df["ScrippsClinic"] = scripps_clinic_algorithm(df, activity_col)
        df["RescoredScrippsClinic"] = webster_rescoring_rules(df["ScrippsClinic"])

        #Apply Sadeh
        df["Sadeh"] = sadeh_algorithm(df, activity_col)
        df["RescoredSadeh"] = webster_rescoring_rules(df["Sadeh"])

        # Apply Oakley
        df["Oakley10"] = oakley_algorithm(df, activity_col, threshold=10)
        df["RescoredOakley10"] = webster_rescoring_rules(df["Oakley10"])

        # Apply Cole-Kripke
        df["ColeKripke"] = cole_kripke_algorithm(df, activity_col)
        df["RescoredColeKripke"] = webster_rescoring_rules(df["ColeKripke"])

        # Apply Sazonov
        df["Sazonov"] = sazonov_algorithm(df, activity_col)
        df["RescoredSazonov"] = webster_rescoring_rules(df["Sazonov"])

        return ["ScrippsClinic", "Sadeh", "Oakley10", "ColeKripke"], ["RescoredScrippsClinic", "RescoredSadeh", "RescoredOakley10", "RescoredColeKripke"]


    # Webster Rescoring Rules 
    def webster_rescoring_rules(s, rescoring_rules="abcde"):

        if s.empty:
            return s

        haveAppliedAnyOtherRule = False

        if "a" in rescoring_rules or "A" in rescoring_rules:
            # After at least 4 minutes scored as wake, next minute scored as sleep is rescored wake
            #print "Processing rule A"
            maskA = s.shift(1).rolling(window=4, center=False, min_periods=1).sum() > 0 # avoid including actual period
            result = s.where(maskA, 0)
            haveAppliedAnyOtherRule = True

        if "b" in rescoring_rules or "B" in rescoring_rules:
            # After at least 10 minutes scored as wake, the next 3 minutes scored as sleep are rescored wake
            #print "Processing rule B"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                s = result

            maskB = s.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
            result = s.where(maskB, 0).where(maskB.shift(1), 0).where(maskB.shift(2), 0)
            haveAppliedAnyOtherRule = True

        if "c" in rescoring_rules or "C" in rescoring_rules:
            # After at least 15 minutes scored as wake, the next 4 minutes scored as sleep are rescored as wake
            #print "Processing rule C"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                s = result

            maskC = s.shift(1).rolling(window=15, center=False, min_periods=1).sum() > 0 # avoid including actual period
            result = s.where(maskC, 0).where(maskC.shift(1), 0).where(maskC.shift(2), 0).where(maskC.shift(3), 0)
            haveAppliedAnyOtherRule = True

        if "d" in rescoring_rules or "D" in rescoring_rules:
            # 6 minutes or less scored as sleep surroundeed by at least 10 minutes (before or after) scored as wake are rescored wake
            #print "Processing rule D"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                s = result

            # First Part
            maskD1 = s.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpD1 = s.where(maskD1.shift(5), 0)
            haveAppliedAnyOtherRule = True

            # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
            maskD2 = s.shift(-10).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpD2 = s.where(maskD2.shift(-5), 0)

            result = tmpD1 & tmpD2

        if "e" in rescoring_rules or "E" in rescoring_rules:
            # 10 minutes or less scored as sleep surrounded by at least 20 minutes (before or after) scored as wake are rescored wake
            #print "Processing rule E"
            if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
                s = result

            # First Part
            maskE1 = s.shift(1).rolling(window=20, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpE1 = s.where(maskE1.shift(9), 0)

            # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
            maskE2 = s.shift(-20).rolling(window=20, center=False, min_periods=1).sum() > 0 # avoid including actual period
            tmpE2 = s.where(maskE2.shift(-9), 0)

            result = tmpE1 & tmpE2

        return result
    
    ### INCLUDE OUR HR- RESCORING HERE?####

    #Scripps Clinic Algorithm Definition
    def scripps_clinic_algorithm(df, activityIdx, scaler = 0.204):
        act_series = {}

        act_series[activityIdx] = df[activityIdx].fillna(0.0)

        # Enrich the dataframe with temporary values
        for i in range(1,11):
            act_series["_a-%d" % (i)] = df[activityIdx].shift(i).fillna(0.0)
            act_series["_a+%d" % (i)] = df[activityIdx].shift(-i).fillna(0.0)

        # Calculates Scripps clinic algorithm
        scripps = scaler * (0.0064 * act_series["_a-10"] + 0.0074 * act_series["_a-9"] +
                        0.0112 * act_series["_a-8"] + 0.0112 * act_series["_a-7"] +
                        0.0118 * act_series["_a-6"] + 0.0118 * act_series["_a-5"] +
                        0.0128 * act_series["_a-4"] + 0.0188 * act_series["_a-3"] +
                        0.0280 * act_series["_a-2"] + 0.0664 * act_series["_a-1"] +
                        0.0300 * act_series[activityIdx] + 0.0112 * act_series["_a+1"] +
                        0.0100 * act_series["_a+2"])


        # Returns a series with binary values: 1 for sleep, 0 for awake
        return (scripps < 1.0).astype(int)


    # %%
    # Sadeh Algorithm
    def sadeh_algorithm(df, activityIdx, min_threshold=0, minNat=50, maxNat=100, window_past = 6, window_nat = 11, window_centered = 11 ):
        """
        Sadeh model for classifying sleep vs active
        """
        _mean = df[activityIdx].rolling(window=window_centered, center=True, min_periods=1).mean()
        _std = df[activityIdx].rolling(window=window_past, min_periods=1).std()
        _nat = ((df[activityIdx] >= minNat) & (df[activityIdx] <= maxNat)).rolling(window=window_nat, center=True, min_periods=1).sum()

        _LocAct = (df[activityIdx] + 1.).apply(np.log)

        sadeh = (7.601 - 0.065 * _mean - 0.056 * _std - 0.0703 * _LocAct - 1.08 * _nat)

        # Returns a series with binary values: 1 for sleep, 0 for awake
        return (sadeh > min_threshold).astype(int)


    # %%
    # Oakley Algorithm
    def oakley_algorithm(df, activityIdx, threshold=80):
        """
        Oakley method to class sleep vs active/awake
        """
        act_series = {}

        act_series[activityIdx] = df[activityIdx].fillna(0.0)
        for i in range(1,5):
            act_series["_a-%d" % (i)] = df[activityIdx].shift(i).fillna(0.0)
            act_series["_a+%d" % (i)] = df[activityIdx].shift(-i).fillna(0.0)

        oakley = 0.04 * act_series["_a-4"] + 0.04 * act_series["_a-3"] + 0.20 * act_series["_a-2"] +                 0.20 * act_series["_a-1"] + 2.0 * act_series[activityIdx] + 0.20 * act_series["_a+1"] +                 0.20 * act_series["_a-2"] + 0.04 * act_series["_a-3"] + 0.04 * act_series["_a-4"]

        return (oakley <= threshold).astype(int)


    # %%
    # Cole Kripke algorithm
    def cole_kripke_algorithm(df, activityIdx):
        """
        Cole-Kripke method to classify sleep vs awake
        """
        act_series = {}

        act_series["_A0"] = df[activityIdx].fillna(0.0)
        for i in range(1,5):
            act_series["_A-%d" % (i)] = df[activityIdx].shift(i).fillna(0.0)
        for i in range(1,3):
            act_series["_A+%d" % (i)] = df[activityIdx].shift(-i).fillna(0.0)

        w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [404, 598, 326, 441, 1408, 508, 350]
        p = 0.00001

        result = p * (w_m4 * act_series["_A-4"] + w_m3 * act_series["_A-3"] + w_m2 * act_series["_A-2"] +                w_m1 * act_series["_A-1"] + w_0 * act_series["_A0"] + w_p1 * act_series["_A+1"] +                w_p2 * act_series["_A+2"])

        return (result < 1.0).astype(int)


    # %%
    # Sazonov algorithm
    def sazonov_algorithm(df, activityIdx):
        """
        Sazonov formula as shown in the original paper
        """
        act_series = {}

        for w in range(1,6):
            act_series["_w%d" % (w-1)] = df[activityIdx].rolling(window=w, min_periods=1).max()

        sazonov = 1.727  - 0.256 * act_series["_w0"] - 0.154 * act_series["_w1"] -                0.136 * act_series["_w2"] - 0.140 * act_series["_w3"] - 0.176 * act_series["_w4"]

        return (sazonov >= 0.5).astype(int)


    # %% THIS NEEDS SOME TWEEKING, WE SHOULD ALSO ADD AN ENSEMBLE THAT IS ON ABSOLUTE AGREEMENT
    # Ensemble model for sleep algorithms
    def sleep_likelihood_by_ensemble(df, cols):
        """
        From a list of sleep metrics (cols), calculate the lihelihood of sleep (i.e, % votes)
        """
        return df[cols].sum(axis=1) / len(cols)

    # In the future include pre-trained ML/DL models here (optimize for features obtained)


