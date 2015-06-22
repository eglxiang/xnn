import h5py
import numpy as np
from HDF5BatchLoad import *

class HDF5RamPool(object):
    def __init__(self, batchReader, partition='train', nBatchInPool=None, refreshPoolProp=None ):
        """
        Define a data loader that aggregates batches from a single partition of an HDF5 file
        :param batchReader: HDF5BatchLoad object set up to extract desired information from HDF5 
        :param partition:  partition from which to load data
        :param nBatchInPool:  number of batches to load into pool initially.  If None, load all batches (might run out of memory)
        :param refreshPoolProp: proportion of pool examples to be refreshed between calls.  If None, pool refreshed entirely
        """
        self.batchReader = batchReader
        self.pool = {}
        self.poolloaded = False
        self.nBatchInPool = nBatchInPool
        self.refreshPoolProp = refreshPoolProp
        self.partition = partition
        self.batchIDlist = []

    def __call__(self):
        self._refreshPool()
        return self.pool
    
    def nInPool(self):
        if len(self.pool)==0 or len(self.pool.keys())==0:
            return 0
        return self.pool[self.pool.keys()[0]].shape[0]

    def _loadPool(self):
        self.currentID = 0
        if self.nBatchInPool is None:
            self.nBatchInPool = self.batchReader.num_batches(self.partition)
        poolitems = []
        self._shuffleIDlist()
        for i in xrange(self.nBatchInPool):
            poolitems.append(self.batchReader(self.partition,batchind=self.batchIDlist[self.currentID+i]))
        self.pool = self._merge_items(poolitems)
        self.currentID += self.nBatchInPool

    def _shuffleIDlist(self):
        self.batchIDlist = np.random.permutation(np.arange(self.batchReader.num_batches(self.partition)))


    def _refreshPool(self):
        if self.refreshPoolProp is None or not self.poolloaded:
            self._loadPool()
            self.poolloaded = True
        elif self.refreshPoolProp > 0. and self.nBatchInPool is not None:
            nInPool = self.nInPool()
            nToRefresh = int(round(nInPool*self.refreshPoolProp))
            numAdded = 0
            while numAdded<nToRefresh:
                b = self.batchReader(self.partition,batchind=self.batchIDlist[self.currentID+1])
                self.currentID += 1
                if self.currentID >= self.batchReader.num_batches(self.partition):
                    self._shuffleIDlist 
                    self.currentID = 0
                numAdded += self._replace_items(b,nToRefresh-numAdded)
        else:
            #no need to refresh
            pass

    def _replace_items(self,items,n):
        nnew = items[items.keys()[0]].shape[0]
        nit = min(n,nnew)
        replaceIDs = np.random.permutation(np.arange(self.nInPool()))[0:nit]
        newIDs = np.random.permutation(np.arange(nnew))[0:nit]
        for key in self.pool.keys():
            self.pool[key][replaceIDs,...] = items[key][newIDs,...]
        return nit

    def _merge_items(self,mergelist):
        if len(mergelist)<1:
            raise RuntimeError('mergelist is empty')
        items = mergelist[0]
        for i in xrange(1,len(mergelist)):
            for key in items.keys():
                items[key] = np.concatenate((items[key],mergelist[i][key]),axis=0)
        return items

    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k in {'batchIDlist','pool'}:
                continue
            if hasattr(self.__dict__[k],'to_dict'):
                properties[k] = self.__dict__[k].to_dict()
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

