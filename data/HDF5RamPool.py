import h5py
import numpy as np
from HDF5BatchLoad import *
import theano

class HDF5RamPool(object):
    """
    Define a data loader that aggregates batches from a single partition of an HDF5 file

    Parameters
    ----------
    batchReader : :class:`HDF5BatchLoad`
        Loader set up to extract desired information from an HDF5 file 
    partition : str
        Partition in HDF5 from which to load data.
    nBatchInPool :  int or None
        Number of batches to load into pool initially.  If None, load all batches from HDF5 (might run out of memory).
    refreshPoolProp : float
        Proportion of pool examples to be refreshed between calls.  Refreshing a pool means loading different examples from the HDF5 file on disk into the pool on RAM.  If None, the pool is refreshed entirely.
    poolSizeToReturn : int
        Number of examples from the pool to be returned when called.  If None, entire pool is returned.
    """
    def __init__(self, batchReader, partition='train', nBatchInPool=None, refreshPoolProp=None, poolSizeToReturn=None ):
        self.batchReader = batchReader
        self.pool = {}
        self.poolloaded = False
        self.nBatchInPool = nBatchInPool
        self.refreshPoolProp = refreshPoolProp
        self.partition = partition
        self.batchIDlist = []
        self.poolSizeToReturn = poolSizeToReturn 

    def __call__(self):
        """
        Returns
        -------
        dict
            The data in the pool.
        """
        self._refreshPool()
        if self.poolSizeToReturn is not None:
            return self._getsubpool()
        return self.pool
   
    def datakeys(self):
        """
        Returns
        -------
        list
            The keys in the dictionary of data in the pool.

        """
        return self.batchReader.datakeys()

    def nInPool(self):
        """
        Returns
        -------
        int
            number of examples currently in the pool.
        """
        if len(self.pool)==0 or len(self.pool.keys())==0:
            return 0
        return self.pool[self.pool.keys()[0]].shape[0]

    def _getsubpool(self):
        subpool = {}
        ids = np.random.choice(self.nInPool(),size=self.poolSizeToReturn)
        for k in self.pool.keys():
            subpool[k] = self.pool[k][ids,...]
        return subpool

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
        keyitems = {k:[] for k in items.keys()}
        for key in items.keys():
            for ml in mergelist:
                keyitems[key].append(ml[key])
        for key in items.keys():
            items[key] = np.concatenate(keyitems[key],axis=0)

#        for i in xrange(1,len(mergelist)):
#            for key in items.keys():
#                items[key] = np.concatenate((items[key],mergelist[i][key]),axis=0)
        return items

    def does_pool_update(self):
        """
        Returns
        -------
        bool
            Whether or not this pool will refresh when it is called.
        """
        if (self.refreshPoolProp == 0 or self.nBatchInPool is None) and self.poolSizeToReturn is None:
            return False
        return True


    def to_dict(self):
        """
        Returns
        -------
        dict
            Dictionary representation of this data pool.
        """
        properties = {}
        for k in self.__dict__:
            if k in {'batchIDlist','pool'}:
                continue
            if hasattr(self.__dict__[k],'to_dict'):
                properties[k] = self.__dict__[k].to_dict()
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

class PoolMerger(object):
    """
    Allows pooling to happen across multiple HDF5 files set up with their own
    :class:`HDF5RamPool` objects.  In the case where keys in the data do not
    overlap, they are filled with NaN values.

    Parameters
    ----------
    poolers : list of :class:`HDF5RamPool`
        The poolers whose data will be merged
    """
    def __init__(self, poolers):
        if isinstance(poolers,list):
            self.poolers = poolers
        else:
            self.poolers = [poolers]
        self.datasizes = None
        self.datakeys  = None
        self.pool = {} 


    def __call__(self):
        """
        Returns
        -------
        dict
            Data pooled from all member poolers.
        """
        self._get_pools()
        if self.datasizes is None:
            self._init_poolinfo()
        if len(self.pool) == 0 or self.anyupdate:
            self._merge_pools()
        return self.pool
        
    def _merge_pools(self):
        pool = {}
        keyitems = {k:[] for k in self.datakeys} 
        for key in keyitems.keys():
            for i,p in enumerate(self.pools):
                if key in p.keys():
                    keyitems[key].append(p[key])
                else:
                    keyitems[key].append(np.nan*np.ones(self.datasizes[key][i],dtype=theano.config.floatX))
        for key in keyitems.keys():
            pool[key] = np.concatenate(keyitems[key],axis=0)
        self.pool = pool

    def _init_poolinfo(self):        
        self.datakeys = set()
        self.datasizes = dict()
        nsinpool=dict()
        sizeinkey= dict()
        self.anyupdate = False
        for plr in self.poolers:
            self.datakeys.update(plr.datakeys())
            self.anyupdate = self.anyupdate or plr.does_pool_update()
        for k in self.datakeys:
            self.datasizes[k] = [] 
            for i,p in enumerate(self.pools):
                if k not in p:
                    self.datasizes[k].append(None)
                else:
                    nsinpool[i] = p[k].shape[0]
                    sizeinkey[k] = p[k].shape[1:]
                    self.datasizes[k].append(p[k].shape)
        for k in self.datakeys:
            for i,p in enumerate(self.pools):
                if k not in p:
                    shp = [nsinpool[i]]
                    shp.extend(sizeinkey[k])
                    shp = tuple(shp)
                    self.datasizes[k][i] = shp 

    def _get_pools(self):
        self.pools = []
        for p in self.poolers:
            self.pools.append(p())

    def nInPool(self):
        """
        Returns
        -------
        int
            number of examples currently in the pool.
        """
        numpool = 0
        for p in self.poolers:
            numpool += p.nInPool()
        return numpool 

    def to_dict(self):
        """
        Returns
        -------
        dict
            A dictionary representation of the pool merger.
        """
        d = {}
        plrs = []
        for p in self.poolers:
            plrs.append(p.to_dict())
        d['poolers'] = plrs
        return d

