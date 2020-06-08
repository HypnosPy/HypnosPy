from hypnospy import Wearable

class SleepWakeAnalysis(object):

    def __init__(self, wearable: Wearable):
        self.wearable = wearable
    
    # Include sleeping window approach here (SLEEP 2020 paper)
    
    def find_sleepwin():
        # Apply our method based on HR here
        
        # if no HR look for expert annotations _anno
        
        # if no annotations look for diaries
        
        # if no diaries apply Van Hees heuristic method
        pass
    
    # Include all Heuristic Traditional Algorithms Here
    def run_sadeh():
        # apply Sadeh
        pass
    
    # In the future include pre-trained ML/DL models here


