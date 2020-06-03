import pandas as pd


class PreProcessing(object):

    def __init__(self, filename=None, device_location=None, additional_data=None):
        self.possible_locations = ["bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all"]

        if filename is not None:
            self.load_file(filename, device_location, additional_data)

    def load_file(self, filename, device_location, additional_data=None):

        if device_location not in self.possible_locations:
            print("ERROR: Device location %s not implmented. Options are %s" % (device_location, ','.join(self.possible_locations)))

        self.device_location = device_location
        self.additional_data = additional_data


    def export_hypnospy(self, filename):
        pass


