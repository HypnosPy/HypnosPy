import pandas as pd
import h5py

class PreProcessing(object):

    def __init__(self, filename=None,
                 collection_name=None,
                 device_location=None,
                 additional_data=None):
        """

        :param filename: input filepath
        :param device_location: where this device was located (options are: "bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all")
        :param additional_data:
        """
        self.possible_locations = ["bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all"]
        self.possible_collections = ["mesa", "latinos", "hchs"]

        if filename is not None:
            self.load_file(filename, collection_name, device_location, additional_data)


    def load_file(self,
                  filename,
                  collection_name,
                  device_location,
                  additional_data=None):

        self.filename = filename
        if device_location not in self.possible_locations:
            print("ERROR: Device location '%s' not implemented. Options are %s" % (device_location, ','.join(self.possible_locations)))

        if collection_name is None or collection_name not in self.possible_collections:
            # TODO: Should we restrict the user like that?
            print("ERROR: Collection '%s' not recognized. Possible collections: %s."% (collection_name, ','.join(self.possible_collections)))

        self.device_location = device_location
        self.additional_data = additional_data
        self.collection_name = collection_name

        self.data = self.__get_wearable_type(self.filename)


    def export_hypnospy(self, filename):
        """

        :param filename:
        :return:
        """

        # TODO: find a better way to save these fields to file
        self.data.to_hdf(filename, key='data', mode='w')
        s = pd.Series([self.device_location, self.collection_name, self.additional_data])
        s.to_hdf(filename, key='other')

        # mydict = dict(data=self.data,
        #               location=self.device_location,
        #               collection=self.collection_name,
        #               additional=self.additional_data)
        #
        # with h5py.File(filename, 'w') as hf:
        #     grp = hf.create_group('alist')
        #     for k, v in mydict.items():
        #         grp.create_dataset(k, data=v)

        #hf.create_dataset('data', data=mydata)

        #hf.create_dataset('location', data=)
        #hf.create_dataset('collection', data=self.collection)
        #hf.create_dataset('additional_data', data=self.additional_data)

        print("Saved file %s." % (filename))


    def __get_wearable_type(self, filename):
        """ Obtain device type
        Used to decide which way to parse the data from input file

        :return: Wearable type (Axivity, GeneActiv, Actigraph, Actiwatch and Apple Watch currently supported)
        """
        f = filename.lower()
        if f.endswith('.cwa') or f.endswith('.cwa.gz') or f.endswith('CWA'):
            return self.__process_axivity(filename)
        elif f.endswith('.bin'):
            return self.__process_geneactiv(filename)
        elif f.endswith('.dat'):
            # here ask for which device: ={ "applewatch":"Apple Watch","apple" [...] "actiwatch"[...]
            # ask if the data is raw or not--> if not--> move down the pecking order
            return self.__process_actigraph(filename)
        elif f.endswith('.csv'):
            return self.__process_csv(filename)
        else:
            print("ERROR: Wearable format not supported for file: " + filename)

    def __process_csv(self, csvfile):
        return pd.read_csv(csvfile)

    def __process_axivity(self, cwaFile):
        """ Use PAMPRO here
        """
        pass

    def __process_actigraph(self, datFile):
        """ Use PAMPRO here
        """
        pass

    def __process_geneactiv(self, datFile):
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

    def obtain_PA_metrics():
        """ Use PAMPRO for derivation of VM, VMHPF, ENMO, Pitch, Roll, Yaw
              Include proxy for MVPA, PA
        """
        # run this if class isn't count based
        # returns something that can be then summarized
        pass

    def obtain_data():
        """ Here we will define epochperiod, columns to be included, and generate
        the new .hpy file
        """
        pass


