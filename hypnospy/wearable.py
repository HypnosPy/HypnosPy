import hypnospy
import pandas as pd
import h5py

class Wearable(object):

    def __init__(self, input):
        """

        :param input: Either a path to a PreProcessing file saved with ``export_hyp`` or a PreProcessing object
        """
        self.data = None

        if type(input) == str:
            # Reads a hypnosys file from disk
            self.__read_hypnospy(input)

        elif type(input) == hypnospy.data.preprocessing.RawProcessing:
            print("Loaded a raw preprocessing object")
            self.__read_preprocessing_obj(input)

    def __read_preprocessing_obj(self, input):
        self.data = input.data
        self.device_location = input.device_location
        self.collection_name = input.collection_name
        self.additional_data = input.additional_data

    def __read_hypnospy(self, filename):
        self.data = pd.read_hdf(filename, 'data')
        l = pd.read_hdf(filename, 'other')
        self.device_location, self.collection_name, self.additional_data = l

        #
        # hf = h5py.File(filename, 'r')
        # self.data = hf.get('data')
        # self.device_location = hf.get('location')
        # self.collection_name = hf.get('collection')
        # self.additional_data = hf.get('additional_data')
