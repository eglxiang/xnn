import h5py
import random
from copy import deepcopy
from HDF5FieldReader import *

class HDF5BatchLoad(object):
    def __init__(self, filepath, inputReaders=[]):
        """
        Define a batch generator that reads batches directly from an HDF5 file
        :param filepath: the path to the hdf5 file
        :param partition: train, test or valid
        :param inputReaders: list of HDF5FieldReader objects that extract input info from HDF5
        """
        self.filepath = filepath
        self.hdf5data = h5py.File(self.filepath, 'r')
        self.inputReaders = []
        self.addReaders(inputReaders)
        self.nb = {}
    
    def __call__(self,partition='train',batchind=None):
        """
        Returns a batch 
        :param batchind: batch index to read.  If None, or if out of range, a random batch will be returned
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
        if type(inputReaders) is list:
            self.inputReaders.extend(inputReaders)
        elif type(inputReaders) is HDF5FieldReader:
            self.inputReaders.append(inputReaders)
   
    def num_batches(self,partition='train'):
        return self._num_batches_in_file(partition)

    def _num_batches_in_file(self,partition='train'):
        """
        Return number of batches in the specified partition of the HDF5 file
        :param partition: train, test or valid
        :return: number of batches
        """
        if partition not in self.nb.keys():
            self.nb[partition] = len(self.hdf5data[partition].keys())

        return self.nb[partition]
    
    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k == 'hdf5data':
                continue
            if k == 'inputReaders':
                properties[k] = [i.to_dict() for i in self.__dict__[k] if hasattr(i,'to_dict')]
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties


def main():
    bl = HDF5BatchLoad('tmp/biggray.hdf5',inputReaders=dgfxReaderList)
    b = bl()


if __name__=="__main__":
    main()


