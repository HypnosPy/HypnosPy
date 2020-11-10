import pandas as pd

class Diary(object):

    def __init__(self):
        """

        :param input: Either a path to a PreProcessing file saved with ``export_hyp`` or a PreProcessing object
        """
        self.data = None

    def from_file(self, file_path):
        self.data = pd.read_csv(file_path)
        if "pid" not in self.data.keys():
            # TODO: decide if we can allow the user to specify it.
            raise KeyError("Diary needs to have a 'pid' column.")

        self.data["pid"] = self.data["pid"].astype(str)
        return self

    def from_dataframe(self, dataframe):
        self.data = dataframe
        return self