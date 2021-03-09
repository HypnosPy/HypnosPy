# Analysis 🔬

## Sleep Annotations 🛏
[diary.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/diary.py)

```python
from hypnospy import Diary
w = Wearable(preprocessed_object)
w.add_diary(Diary().from_file(diary_file))
```

## Individual Analysis 🗓

### **Sleep Labelling**

```python
from hypnospy import Wearable
from hypnospy.data import MESAPreProcessing
from hypnospy.analysis import SleepWakeAnalysis, SleepMetrics, SleepBoudaryDetector

# MESAPreProcessing is a specialized class to preprocess csv files from Philips Actiwatch Spectrum devices used in the MESA Sleep experiment
# MESA Sleep dataset can be found here: https://sleepdata.org/datasets/mesa/
preprocessed = MESAPreProcessing("../data/examples_mesa/mesa-sample.csv")

# Wearable is the main object in HypnosPy.
w = Wearable(preprocessed)

# In HypnosPy, we have the concept of ``experiment day'' which by default starts at midnight (00 hours).
# We can easily change it to any other time we wish. For example, lets run this script with experiment days
# that start at 3pm (15h)
w.change_start_hour_for_experiment_day(15)

# Sleep Wake Analysis module
sw = SleepWakeAnalysis(w)
sw.run_sleep_algorithm("ScrippsClinic", inplace=True) # runs alg and creates new col named 'ScrippsClinic'
sw.run_sleep_algorithm("Cole-Kripke", inplace=True)   # runs alg and creates new col named 'Cole-Kripke'

# Detect the boundaries of sleep labels 
sb = SleepBounaryDetector(w)
sb.detect_sleep_boundaries(strategy, output_col: "hyp_sleep_period",
                                annotation_hour_to_start_search: 18, annotation_col: None,
                                annotation_merge_tolerance_in_minutes: 20,
                                annotation_only_largest_sleep_period: True,
                                hr_quantile: 0.4, hr_volarity_threshold: 5,
                                hr_rolling_win_in_minutes: 5, 
                                hr_sleep_search_window: (20, 12),
                                hr_min_window_length_in_minutes: 40, 
                                hr_volatility_window_in_minutes: 10,
                                hr_merge_blocks_gap_time_in_min: 240,
                                hr_sleep_only_in_sleep_search_window: False,
                                hr_only_largest_sleep_period: False,
                                angle_cols: list = [],
                                angle_use_triaxial_activity: False, 
                                angle_start_hour: int = 15,
                                angle_quantile: float = 0.1, 
                                angle_minimum_len_in_minutes: 30,
                                angle_merge_tolerance_in_minutes: 180,
                                angle_only_largest_sleep_period: True)

# Calculate and evaluate sleep metrics against ground truth. 
sm = SleepMetrics(w)
sm.calculate_sleep_efficiency(sm.data, sleep_wake_col: 'hyp_sleep_period', ignore_awakenings_smaller_than_X_epochs: 5)
sm.evaluate_sleep_metric(ground_truth: Diary(w).from_file(diary_file), sleep_wake_col: 'hyp_sleep_period')
```

[sleep_boundary_detector.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/sleep_boundary_detector.py)\

**SleepBoundaryDetector** labels the sleep-wake periods on the Wearable.data dataframe using any of the following strategies:
* HR-based algorithm - HR-only signals
* Angle-algorithm - accelerometer-only signals
* Annotation-only - diaries or expert labels

Parameters used:
* *hr_quantile* - looks below that quantile in the daily heart rate ECDF (empirical cumulative distribution function) - specific to subject and day / time period
* *hr_volarity_threshold* - volatility threshold (in beats per minute) for refining the initial sleep boundary labels from the quantile step
* *hr_rolling_win_in_minutes* - number of minutes over which to apply the HR rolling average 
* *hr_sleep_search_window* - adjusts the start and end time when looking for sleep boundaries. Eg. (20,12) will only look betwee 8pm and 12 noon
* *hr_min_window_length_in_minutes* - threshold duration for considering an initial sleep-labelled period as sleep. Eg. if 40, then all periods shorter thna 40 min are ignored
* *hr_volatility_window_in_minutes* - how many minutes to use when calculating HR volatility (epochs for standard deviation)
* *hr_merge_blocks_gap_time_in_min* - if two sleep blocks are separated by less than this period, then they are merged into a single sleep blocks
* *hr_sleep_only_in_sleep_search_window* - only look for sleep within search window (boundaries can not be outside pre-defined times)
* *hr_only_largest_sleep_period* - only keeps longest daily sleep period, if 'False', then naps can be detected too\
* *angle_cols* - columns where the accelerometry data is stored
* *angle_start_hour* - time to start each 24-hour analysis
* *angle-quantile* - threshold of activity below which the angle algorithm initially labels sleep
* *angle_minimum_len_in_minutes* - threshold duration for initial sleep period labels
* *angle_merge_tolerance_in_minutes* - if two sleep blocks are separated by less than this period, then they are merged into a single sleep blocks
* *angle_only_largest_sleep_period* - only keeps longest daily sleep period, if 'False', then naps can be detected too

The angle algorithm was developed by [van Hess et al. (2018)](https://www.nature.com/articles/s41598-018-31266-z)


[sleep_wake_analysis.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/sleep_wake_analysis.py)\
[sleep_metrics.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/sleep_metrics.py)\

Sleep metrics implemented:
* **Sleep efficiency** - 0-1 - (total sleep time - duration of awakenings) / total sleep time
* **Awakenings** - int - number of awakenings longer than x epochs during each night
* **SRI (Sleep Regularity Index)** - percent - indicates how regular one's main sleep windows are, from 0 (sleeps at random times each days) to 100 (sleep between the same interval each day) - metric developed in [this paper](https://www.nature.com/articles/s41598-017-03171-4)
* **Total Sleep Time** - hours - duration of main labelled sleep window
* **Total Time in Bed** - Total Sleep Time + Sedentary Time detected before and after main sleep window
* **Total Wake Time** - total duration of awakenings during each main sleep window

If sleep annotations such as polysomnography-derived expert labels or subject-annotated sleep diaries are available, they can serve as
ground truth for evaluating the performance of HypnosPy in terms of the usual evaluation metrics:
* *Accuracy*
* *Precision*
* *Recall*
* *F1_score*
* *ROC_AUC*
* *Cohen's kappa*

### **Circadian Analysis**

[circadian_analysis.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/circadian_analysis.py)

### **Physical Activity Analysis**

[physical_activity.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/phyisical_activity.py)
[non_wearing_detector.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/non_wearing_detector.py)

### **Validation**
[validator.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/validator.py)

## Visualization 🖥

[visualization.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/analysis/visualization.py)

## Population Analysis

[experiment.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/experiment.py)
