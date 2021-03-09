# Preprocessing ⚗
The first step in the HypnosPy workflow is creating the **RawProcessing()** class for each subject's data:

[preprocessing.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/data/preprocessing.py)

```python
from hypnospy.data import RawProcessing
```

This takes the follwing parameters:
* filename - a string containing the file name
* cols_for_activity - 
* col_for_mets - column name for pre-existing MET data
* is_enmo - default: False. True if the cols_for_activity are already calculated as the ENMO (Euclidean Norm Minus One)
* is_act_count - default: False. True is the cols_for_activity are already the activity counts
* col_for_datetime - column where timestamp is present
* start_of_week - define integer of first weekday
* strftime - string with datetime format of 'col_for_datetime' column
* col_for_pid - column name with participant's ID 
* pid - sets the participant's ID in the study
* additional_data - object containing any additional data for input
* device_location - body location where the subject wore the device (eg. dominant wrist)

The following classes are built upon **RawProcessing** to analyse data from specific studies:
```python
from hypnospy.data import MESAPreProcessing
from hypnospy.data import MMASHPreProcessing
from hypnospy.data import HCHSPreProcessing
from hypnospy.data import ActiwatchSleepData
```

The next step in the workflow is calling **Wearable()** as child class of the preprocessing objects. A previously-saved file (via the export_hypnospy method of **RawProcessing()**) can also be used.
**Wearable()** is the main HypnosPy object.

[wearable.py](https://github.com/HypnosPy/HypnosPy/blob/master/hypnospy/wearable.py)

```python
from hypnospy import Wearable
```
Wearable's essential key is 'data', which stores all the subject's signals as a timeseries. For example
```python echo=TRUE
from hypnospy.data import MESAPreProcessing
from hypnospy import Wearable
preprocessed = MESAPreProcessing("../data/examples_mesa/mesa-sample.csv")
w = Wearable(preprocessed)
print(w.data.head())
```
