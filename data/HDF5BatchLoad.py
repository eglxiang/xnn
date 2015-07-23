import h5py
import random
from copy import deepcopy
from HDF5FieldReader import *

class HDF5BatchLoad(object):
    def __init__(self, filepath, inputReaders=[]):
        """
        Define a batch generator that reads batches directly from an HDF5 file.
        The HDF5 file must be structured with the groups
        'train','test','valid'.  Each of these groups must contain a group for
        each batch, with the group name 'batchN' where N is an integer in the
        range 0 to the number of batches.  Each batch group must have datasets
        inside it that contain the data of interest.

        Parameters
        ----------

        filepath : str
            The path to the hdf5 file.
        partition : str
            'train', 'test', or 'valid'
        inputReaders : list of :class:`HDF5FieldReader` 
            Readers that extract input from the HDF5 file
        """
        self.filepath = filepath
        self.hdf5data = h5py.File(self.filepath, 'r')
        self.inputReaders = []
        self.addReaders(inputReaders)
        self.nb = {}
    
    def __call__(self,partition='train',batchind=None):
        """
        Returns the data from a single batch group from the HDF5 data

        Parameters
        ----------

        batchind : int
            batch index to read.  If None, or if out of range, a random batch
            will be returned

        Returns
        -------
        dict
            Dictionary with keys corresponding to the names of the :class:`HDF5FieldReader` objects and values corresponding to the data as read and preprocessed by the :class:`HDF5FieldReader` objects
        """
        nb = self.num_batches(partition)
        if nb == 0:
            print("Asking for a batch from empty partition %s from file: %s"%(self.partition,self.filepath)) 
            return {}

        if batchind is None or batchind > nb:
            batchind = random.randint(0,nb-1)
        batchdata = self.hdf5data[partition]['batch%d' % batchind]
        items = {}
        for inpRead in self.inputReaders:
            inpname = inpRead.getName()
            inpVal  = inpRead.readBatch(batchdata)
            items[inpname]=inpVal
        return items

    def addReaders(self,inputReaders):
        """
        Add a list of :class:`HDF5FieldReader` objects to the loader.

        Parameters
        ----------
        inputReaders : list
            list of :class:`HDF5FieldReader` objects to add to the batch loader.
        """
        if type(inputReaders) is list:
            self.inputReaders.extend(inputReaders)
        elif type(inputReaders) is HDF5FieldReader:
            self.inputReaders.append(inputReaders)
   
    def num_batches(self,partition='train'):
        """
        Get the number of batches in a partition in the HDF5 file

        Parameters
        ----------

        partition : str
            The partition in the HDF5 file for which to retrieve the number of
            batches.

        Returns
        -------
        int
            Number of batch groups inside :py:attr:`partition`
        """
        return self._num_batches_in_file(partition)

    def _num_batches_in_file(self,partition='train'):
        if partition not in self.nb.keys():
            self.nb[partition] = len(self.hdf5data[partition].keys())

        return self.nb[partition]
    
    def to_dict(self):
        """
        Serialize to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the :class:`HDF5BatchLoad`
        """
        properties = {}
        for k in self.__dict__:
            if k == 'hdf5data':
                continue
            if k == 'inputReaders':
                properties[k] = [i.to_dict() for i in self.__dict__[k] if hasattr(i,'to_dict')]
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

    def datakeys(self):
        """
        Get a list of keys that the data returned by this loader will contain.

        Returns
        -------
        list of str
            Keys that the data returned by this loader will contain.

        """
        keys = {i.name for i in self.inputReaders}
        return keys


def main():
    bl = HDF5BatchLoad('tmp/biggray.hdf5',inputReaders=dgfxReaderList)
    b = bl()


if __name__=="__main__":
    main()


