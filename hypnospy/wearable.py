

class Wearable(object):

    def __init__(self, filename, collection_name):

        self.collection_name = collection_name
        self.data = self.__open_hypnospy_file(filename)

    def __open_hypnospy_file(filename):
        print("NOT YET IMPLEMENTED")


