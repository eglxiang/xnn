import numpy as np
import theano

class Weighter(object):
    def __init__(self,labelKey,statPool=None):
        self.statPool = statPool
        self.labelKey = labelKey
        if self.statPool is not None:
            self._init_stats()

    def _init_stats(self):
        pass

    def __call__(self,data):
        return np.ones((data[self.labelKey].shape[0],1)).astype(theano.config.floatX)
        
    def to_dict(self):
        properties = {}
        properties['weightType'] = 'None'
        for k in self.__dict__:
            if k == 'statPool':
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties



class BinnedWeighter(Weighter):

    def __init__(self,labelKey,bins,statPool=None):
        self.bins = bins
        super(BinnedWeighter,self).__init__(labelKey,statPool)        

    def _init_stats(self):
        labels = self.statPool[self.labelKey]
        binlabels = np.digitize(labels[~np.isnan(labels)],bins=self.bins)
        bincounts = np.bincount(binlabels,minlength=len(self.bins))
        bincounts = bincounts[0:len(self.bins)]
        self.bincount = bincounts 
        # smooth the bin counts
        if 0 in self.bincount:
            #TODO: make this smoothing dependent on data?
            self.bincount += .25
        self.binweight = 1./self.bincount

    def __call__(self,data):
        if self.statPool is None:
            self.statPool = data
            self._init_stats()
            self.statPool = None
        labels = data[self.labelKey]
        binlabels = np.digitize(labels.flatten(), bins=self.bins)
        weightsfunc = lambda x: self.binweight[x] if x < len(self.bins) else 0
        weights = np.array([weightsfunc(binlabel) for binlabel in binlabels])
        weights = weights.reshape(-1,1)
        weights[np.isnan(labels)] = 0
        return weights

    def to_dict(self):
        properties = {}
        properties['weightType'] = 'Binned'
        for k in self.__dict__:
            if k == 'statPool':
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

        

class CategoricalWeighter(Weighter):
    def __init__(self,labelKey,statPool=None):
        super(CategoricalWeighter,self).__init__(labelKey,statPool)        

    def _init_stats(self):
        labels = self.statPool[self.labelKey] 
        self.frequencies = np.nansum(labels, axis=0, keepdims=True)
        #if 0 in self.frequencies:
        self.frequencies += .1 
        self.proportions = (1./self.frequencies).astype(theano.config.floatX)

    def __call__(self,data):
        if self.statPool is None:
            self.statPool = data
            self._init_stats()
            self.statPool = None
        labels = data[self.labelKey]
        return labels.dot(self.proportions.T)
    
    def to_dict(self):
        properties = {}
        properties['weightType'] = 'Categorical'
        for k in self.__dict__:
            if k == 'statPool':
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

class BinaryWeighter(Weighter):
    def __init__(self,labelKey,statPool=None):
        super(BinaryWeighter,self).__init__(labelKey,statPool)        
    
    def _init_stats(self):
        lab = self.statPool[self.labelKey]
        numPos = np.nansum(lab, axis=0, keepdims=True)
        numNeg = np.nansum(1-lab, axis=0, keepdims=True)
        meanNum = numPos+numNeg/2
        self.posWeight = meanNum / (numPos) 
        self.negWeight = meanNum / (numNeg)
        self.numNeg = numNeg
        self.numPos = numPos

    def __call__(self,data):
        if self.statPool is None:
            self.statPool = data
            self._init_stats()
            self.statPool = None

        labels = data[self.labelKey]
        weights = np.ones_like(labels)
        weights[labels==1] = self.posWeight
        weights[labels==0] = self.negWeight
        return weights

    def to_dict(self):
        properties = {}
        properties['weightType'] = 'Binary'
        for k in self.__dict__:
            if k == 'statPool':
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties
