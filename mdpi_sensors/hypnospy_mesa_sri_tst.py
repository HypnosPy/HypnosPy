# # In this notebook...
#
# we used the MESA cohort to show how hypnosPy can be used to classify sleep using expert annotations and showcase how sleep metrics can be derived from these annotations as well. 
#
# In one particular example participant, we show that their sleep regularity is poor and total sleep time in two out of the five nights is short.
#
# Further, we show, at a population level how HypnosPy can be used to analyze the association of TST and SRI and even cluster individuals based on SRI levels. 
#
# These type of analyses on SRI in a large population study help confirm previous findings reported in the [literature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6154967/). HypnosPy allows for a variety of other analyses and visualizations given different input modalities.  
#

from glob import glob
from hypnospy import Wearable, Experiment
from hypnospy.data import MESAPreProcessing
from hypnospy.analysis import Viewer, NonWearingDetector, SleepMetrics, SleepBoudaryDetector

# +
# We load all MESA sleep files from disk. These are the original files from sleepdata.org.
files = glob("../data/examples_mesa/actigraphy/mesa-sleep-*.csv")
exp = Experiment()
for file in files[:100]:  # To speed up this experiment, we will only load the first 100 files
    # MESAPreProcessing is a specialized class to preprocess devices from the MESA Sleep collection
    # A column 'hyp_annotation' is automatically created based on annotations of sleep from MESA Sleep
    # (for more details see package documentation, i.e., online or running ``MESAPreProcessing?'' on a terminal)
    preprocessed = MESAPreProcessing(file)

    # Wearables are the main object in Hypnospy.
    # Here we shift the start hour of our experiment day to 3pm
    # And take advantage of the Experiment class to process multiple wearables at same time
    w = Wearable(preprocessed)
    w.change_start_hour_for_experiment_day(15)
    exp.add_wearable(w)

# HypnosPy is able to infer the data frequency, but the user can modify it with this command:
exp.set_freq_in_secs(30)
print("Loaded %d objects" % (len(exp.get_all_wearables())))
# -

# Once more, we remove the non wearing days with this set of commands:
nwd = NonWearingDetector(exp)
nwd.detect_non_wear(strategy="choi")
nwd.check_valid_days(max_non_wear_minutes_per_day=180)
nwd.drop_invalid_days()

# And annotate sleep boundaries using the default annotations from MESA dataset:
sbd = SleepBoudaryDetector(exp)
sbd.detect_sleep_boundaries(strategy="annotation", annotation_col="hyp_annotation",
                            output_col="SleepBoundariesFromAnnotations", annotation_merge_tolerance_in_minutes=30)


# We pick a random wearable to visualize what was done so far:
v = Viewer(exp.get_wearable("288"))
v.view_signals(signal_categories=["activity"], signal_as_area=["SleepBoundariesFromAnnotations"],
               colors={"area": ["green"]}, alphas={"area": 0.5})


# Calculate a few sleep metrics for the whole population
sm = SleepMetrics(exp)
sri = sm.get_sleep_quality("hyp_annotation", metric="sri")
tst = sm.get_sleep_quality("hyp_annotation", metric="totalTimeInBed", sleep_period_col="SleepBoundariesFromAnnotations")
# And plot these sleep metrics to investigate tendencies in dataset:
v.plot_two_sleep_metrics(sri, tst, label_a="Sleep Regularity Index", label_b="Total Sleep Time", color="red", alpha=0.2)
