import pandas as pd


class RawProcessing(object):

    def __init__(self, filename=None, device_location=None, additional_data=None):
        self.possible_locations = ["bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all"]

        if filename is not None:
            self.load_file(filename, device_location, additional_data)

    def load_file(self, filename, device_location, additional_data=None):

        if device_location not in self.possible_locations:
            print("ERROR: Device location %s not implmented. Options are %s" % (device_location, ','.join(self.possible_locations)))

        self.device_location = device_location
        self.additional_data = additional_data


    
    def get_wearable_type(self):
        """ Obtain device type 
        Used to decide which way to parse the data from input file
        
        
        :return: Wearable type (Axivity, GeneActiv, Actigraph, Actiwatch and Apple Watch currently supported)
        """"
        if file.lower().endsiwth('.cwa') or file.lower().endswith('.cwa.gz') or file.lower().endswith('CWA'):
            return process_axivity(file)
        elif file.lower().endswith('.bin'):
            return process_geneactiv(file)
        elif file.lower().endswith('.dat'):
            return process_actigraph(file)
        elif file.lower().endswith('.csv'):
            return 
        # here ask for which device: ={ "applewatch":"Apple Watch","apple" [...] "actiwatch"[...]
        # ask if the data is raw or not--> if not--> move down the pecking order
        elif file.lower().endswith('.dta'):
            return 
        # here ask for which device: ={ "applewatch":"Apple Watch","apple" [...] "actiwatch"[...]
        # ask if the data is raw or not--> if not--> move down the pecking order
        else:
        print("ERROR: Wearable format not supported for file: " + file)
        
    
    def process_axivity(cwaFile):
        """ Use PAMPRO here
        """
        pass
    
    def process_actigraph(datFile):
        """ Use PAMPRO here
        """
        pass
   
    def process_geneactiv(datFile):
        """ Use PAMPRO here
        """
        pass
    
    def run_nonwear():
        """ Use PAMPRO autocalibration
        """
        pass
    def calibrate_data():
        """ Use PAMPRO autocalibration
        """
        pass
    
    def obtain_PA_metrics()
        """" Use PAMPRO for derivation of VM, VMHPF, ENMO, Pitch, Roll, Yaw
              Include proxy for MVPA, PA
        """"
        # run this if class isn't count based 
        # returns something that can be then summarized
        pass
  
  class dataReader():
        r""" Reads processed raw files 
        and files obtained that are already proceessed"
        
        Params
        ---------
        
        """
        
        def __init__(self, wearable_type, datareader=[]):
            
            #store wearable type
            self.__wearable_type = wearable_type
            
            # store list of data reader
            self.__datareader = datareader
        
        @property
        def wearable_type(self):
            r""" This function will take in the type of reader"""
            #1. Check file type from above processing
            
            #2,. 
        
        return 
        
        @property
        def datareader(self):
            r""" This function will """
            #1. Check file type 
            
            #2. Determine  
        
        return 
    
    def find_sleepwin():
        # Apply our method based on HR here
        
        # if no HR look for expert annotations _anno
        
        # if no annotations look for diaries
        
        # if no diaries apply Van Hees heuristic method

    def obtain_data():
        """ Here we will define epochperiod, columns to be included, and generate
        the new .hpy file
        """
        pass
    
    def export_hypnospy(self, filename):
        pass


