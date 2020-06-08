

class TimeSeriesProcessing(object):

    def __init__(self):
        pass
    def data_load(self):
        """ Here we need to load the data and determine:
        potentially by fetching from class wearable
        (1) what type of file is it
        (2) is it multimodal
        (3) length/type- night only/ full
        (4) sampling rate
        """
    pass

    def featurize(self):
        """ Uses tsfresh to extract time series features
        """
        pass
        
    def day_night_split(self):
       """ Splits day night into 9AM to 9PM chunks (day) and 9PM to 9AM (night) (saves flags as columns)
        
       """
       pass
       # do time series and formatting transformations, etc.

      
    def detect_sleep_boundaries(self, strategy):
        """
            Detected the sleep boundaries.

            param
            -----

            strategy: "hr"
        """
    
        # Include HR sleeping window approach here (SLEEP 2020 paper)
        
        # if no HR look for expert annotations _anno
        
        # if no annotations look for diaries
        
        # if no diaries apply Crespo (periods of innactivity)
        
        # if no diaries apply Van Hees heuristic method
        # this method requires triaxial accelerometry to be used
    pass





