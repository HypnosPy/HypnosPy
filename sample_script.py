from hypnospy import PreProcessing, TimeSeriesProcessing, SleepWakeAnalysis, Wearable

# Preprocessing...
pp = PreProcessing()
pp.load_file("dummy.dta", "dw")

#If trixial→ collapse ENMO to (15’’,30’’)
#Determine sampling rate (15’’, 30’’, 1’) → if not, ERROR (‘Device sampling rate not supported’)
pp.export_hypnospy("dummy.hpy") # -> [ typeOfDevice (triaxial, hr, counts), typeOfStudy(full, night_only), location(dw,ndw,hip,chest,bw,bw_ch,bw_hp,hp_ch,all), additional(diary,anno,PSGlabel), df={ _pid, _time, _acc, _hr?, "PSGLabel"} ]
#Pampro -> ourformat.hd5

