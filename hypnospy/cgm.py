# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class CGM(object):
    """
    
    For Freestyle Libre, the imported .csv files have the following descriptions:
            'Serial Number' ID of the row.

            'Device Timestamp' Date and time that indicates when the record was taken.

            'Register Type' column. The type of registers can take the following values:

                0: automatic glucose value register, saved each 15 minutes by the device.
                'Historic Glucose mg/dL' column. Blood glucose value in rows with register type 0 (mg/dl).

                1: manual blood glucose value register, saved in the record after a read by the patient.
                'Scan Glucose mg/dL' column. Blood glucose value in rows with register type 1 (mg/dl).

                2: register of insulin without a numeric value.
                Rapid insulin register without a numeric value in rows with register type 2.
                
                3: register of carbohydrates without a numeric value.
                Carbohydrates without a numeric value in rows with register type 3.

                4: register of insulin done with a numeric value.
                Units of rapid insulin entered by the patient in rows with register type 4.
                
                5: register of carbohydrates with a numeric value.
                Units of carbohydrates entered by the patient in rows with register type 5.
        
        """
    def __init__(self):
        """

        :param input: Either a path to a PreProcessing file saved with ``export_hyp`` or a PreProcessing object
        
        
        """
        
        self.data = None

    def from_file(self, 
                  file_path,
                  pid: str = -1,
                  device_col: str = 'Device',
                  device_serial: str = 'Serial Number',
                  cgm_time_col: str = 'Device Timestamp',
                  strftime: str = '%m-%d-%Y %I:%M %p',
                  reading_type_col: str = 'Record Type',
                  glucose_col_auto: str = 'Historic Glucose mg/dL',
                  glucose_col_man: str = 'Scan Glucose mg/dL',
                  ket_col: str = 'Ketone mmol/L'
                  ):
        """
        

        Parameters
        ----------
        file_path : str
            Path to file.
        pid : str, optional
            DESCRIPTION. The default is -1.
        device_col : str, optional
            DESCRIPTION. The default is 'Device'.
        device_serial : str, optional
            DESCRIPTION. The default is 'Serial Number'.
        cgm_time_col : str, optional
            DESCRIPTION. The default is 'Device Timestamp'.
        strftime : str, optional
            DESCRIPTION. The default is '%m-%d-%Y %I:%M %p'. Time format for device data.
        reading_type_col : str, optional
            DESCRIPTION. The default is 'Record Type'. What is recorded - manual / automatic glucose, insulin dose, food
        glucose_col_auto : str, optional
            DESCRIPTION. The default is 'Historic Glucose mg/dL'. CGM readings
        glucose_col_man : str, optional
            DESCRIPTION. The default is 'Scan Glucose mg/dL'. Manual input of finger strip glucose
        ket_col : str, optional
            DESCRIPTION. The default is 'Ketone mmol/L'. CGM ketone level reading.


        Returns
        -------
        DataFrame
            DESCRIPTION. Contains CGM device metadata, timestamp column 'hyp_time_col' and glucose and ketone readings

        """
        self.data = pd.read_csv(file_path,header=1)
        #if "pid" not in self.data.keys():
            # TODO: decide if we can allow the user to specify it.
        #    raise KeyError("Diary needs to have a 'pid' column.")
        if pid is not None:
            self.data['pid'] = pid
        self.data['device'] = self.data[device_col].astype('str')
        self.data['serial'] = self.data[device_serial].astype('str')
        self.data['hyp_time_col'] = pd.to_datetime(self.data[cgm_time_col],format=strftime)
        self.data.set_index('hyp_time_col', drop=False,inplace=True)
        self.data['auto0_man1'] = self.data[reading_type_col]
        self.data['gluc_mgdl'] = self.data[glucose_col_auto].fillna(0)+self.data[glucose_col_man].fillna(0)
        self.data['ket_mmol'] = self.data[ket_col]
        # Columns with other inputs will be implemented
        self.data = self.data.drop(columns=['Non-numeric Rapid-Acting Insulin', 'Rapid-Acting Insulin (units)',
                                            'Non-numeric Food', 'Carbohydrates (grams)', 'Carbohydrates (servings)',
                                            'Non-numeric Long-Acting Insulin','Long-Acting Insulin (units)',
                                            'Notes', 'Strip Glucose mg/dL','Meal Insulin (units)',
                                            'Correction Insulin (units)','User Change Insulin (units)'])
        self.data = self.data.drop(columns=[device_col, device_serial,cgm_time_col, 
                                            glucose_col_auto,glucose_col_man, reading_type_col, ket_col])
        return self

    def from_dataframe(self, dataframe):
        """
        See cgm.from_file(). Use this with pre-prepared DataFrame.

        """
        self.data = dataframe
        return self
