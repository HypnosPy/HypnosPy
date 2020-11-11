
<img src ="docs/HypnosPy.png" width = "750" class ="center" >

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg)](https://www.python.org/)
[![PyPI download month](https://img.shields.io/pypi/dm/hypnospy.svg)](https://pypi.python.org/pypi/hypnospy/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/hypnospy.svg)](https://pypi.python.org/pypi/hypnospy/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/ippozuelo/HypnosPy/graphs/commit-activity)
[![GitHub watchers](https://img.shields.io/github/watchers/ippozuelo/HypnosPy?style=social&label=Watch&maxAge=2592000)](https://GitHub.com/ippozuelo/HypnosPy/watchers/)
[![GitHub stars](https://img.shields.io/github/stars/ippozuelo/HypnosPy?style=social&label=Star&maxAge=2592000)](https://GitHub.com/ippozuelo/HypnosPy/stargazers/)

# HypnosPy :sleeping_bed:
A Device-Agnostic, Open-Source Python Software for Wearable Circadian Rhythm and Sleep Analysis and Visualization


# Installation :computer:

You can install HypnosPy with pip in a (bash) shell environment, type:

```
pip install hypnospy
```
To update HypnosPy type:

```
pip install -U hypnospy
```

Dependencies include python 3.7 and the following packages:


# Usage :bulb:
Here is a simple example of how you can use HypnosPy in your research:
```python
from hypnospy import Wearable
from hypnospy.data import MESAPreProcessing
from hypnospy.analysis import SleepWakeAnalysis, Viewer, NonWearingDetector

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

# View results
v = Viewer(w)
v.view_signals(signal_categories=["activity"], signal_as_area=["ScrippsClinic", "Cole-Kripke", "Oakley"],
               colors={"area": ["green", "red", "blue"]}, alphas={"area": 0.6})

# Easily remove non-wearing epochs/days.
nwd = NonWearingDetector(w)
nwd.detect_non_wear(strategy="choi")
nwd.check_valid_days(max_non_wear_minutes_per_day=180)
nwd.drop_invalid_days()

```
Some of the amazing features of HypnosPy are shown in the [here](https://github.com/ippozuelo/HypnosPy/blob/master/mdpi_sensors/).
Try it out! :test_tube:


# Under the hood :mag_right:

Here we'll iput a breakdown of the software architecture

<p style="text-align:center;"><img src ="docs/SoftwareArchitecture.png" width = "550" alt="centered image"></p>

Ignacio to provide a breakdown of the main software functionalities here

Circadian

<p style="text-align:center;"><img src ="docs/Circadian.png" width = "550" alt="centered image"></p>


HR algorithm (update)
<p style="text-align:center;"><img src ="docs/HRdescriptioncropped.png" width = "650" alt="centered image"></p>

We found that HR quantiles offered a personalized method to direct our sleeping window search as observed in the figure bellow:
<p style="text-align:center;"><img src ="docs/HRCDF.png" width = "550" alt="centered image"></p>

Example

<p style="text-align:center;"><img src ="docs/examplesubject.png" width = "550" alt="centered image"></p>



# Cite our work! :memo::pencil:

# Contributing :handshake:
We are very keen on having other colleagues contribute to our work and to make this as generalizable as possible of a package.
This project came about due to the frustration of not having a centralized analysis tool that worked across devices, so if you
find our project interesting or think you can improve it, please contribute by:

* reporting bugs (how you got it and if possible, how to solve it)
* adding new tools- if you are interested on this please email one of the main developers, we'd love to hear from you
* adding pre-processing pipelines for new devices. The more, the merrier.
* sharing our work with your colleagues, this will allow the project to improve and not die in this corner of the interweb.
* reaching out!- we are always keen on learning more of how you are using/want to use hypnospy


### License :clipboard:
This project is released under a BSD 2-Clause Licence (see LICENCE file)
### Contributions :man_technologist: :woman_technologist:
* **João Palotti (MIT)** @joaopalotti *main developer*
* **Marius Posa (Cambridge)** @marius-posa *main developer*
* **Ignacio Perez-Pozuelo (Cambridge)** @ippozuelo *main developer*
# Research that uses HypnosPy :rocket:

* Perez-Pozuelo, I., Posa, M., Spathis, D., Westgate, K., Wareham, N., Mascolo, C., ... & Palotti, J. (2020). Detecting sleep in free-living conditions without sleep-diaries: a device-agnostic, wearable heart rate sensing approach. medRxiv.

# Acknowledgements :pray:

* We thank the MRC Epidemiology Unit at Cambridge for supporting some of the research associated to this work as well as QCRI and MIT.

