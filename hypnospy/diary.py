import pandas as pd

class Diary(object):

    def __init__(self):
        """

        :param input: Either a path to a PreProcessing file saved with ``export_hyp`` or a PreProcessing object
        """
        self.data = None

    def from_file(self, file_path):
        self.data = pd.read_csv(file_path)
        return self

    def from_dataframe(self, dataframe):
        self.data = dataframe
        return self