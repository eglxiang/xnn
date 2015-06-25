import numpy as np
from copy import deepcopy
import theano

class Sampler(object):
    def __init__(self, pooler,keysamplers=[],samplemethod='uniform',batchsize=128, numbatches=None, nanOthers=False):
        self.POSSIBLE_METHODS = {'uniform','balance','sequential'}
        self.pooler = pooler
        self.batchsize=batchsize
        self.numbatches=numbatches
        self.keysamplers= keysamplers
        self.keylist = [k.labelKey for k in self.keysamplers]
        self.nanOthers = nanOthers
        self.samplemethod = samplemethod
        if samplemethod not in self.POSSIBLE_METHODS:
            raise NotImplementedError(("samplemethod should be one of: %s"%', '.join(self.POSSIBLE_METHODS)))

    def __call__(self):
        if self.numbatches is None:
            pool = self.pooler()
            self.numbatches = self.pooler.nInPool()//self.batchsize
        for i in xrange(self.numbatches):
            pool = self.pooler()
            if self.samplemethod == 'balance' and len(self.keysamplers)>0:
                batchinds,keyids = self._samplebalanced(pool)
            elif self.samplemethod == 'uniform':
                batchinds,keyids = self._sampleuniform(pool)
            else:
                batchinds,keyids = self._samplesequential(i)
            batch = self._extractInds(pool,batchinds,keyids)
            yield batch

    def _samplesequential(self,i):
        startid = i*self.batchsize
        endid = startid+self.batchsize
        batchinds = map(lambda(x):x%self.pooler.nInPool(),range(startid,endid))
        return batchinds,[]
        

    def _sampleuniform(self,pool):
        batchids = np.random.choice(np.arange(self.pooler.nInPool()),self.batchsize)
        return batchids,[]
        

    def _samplebalanced(self,pool):
        keyids = np.random.choice(np.arange(len(self.keylist)),self.batchsize)
        allbatchids = []
        for ki in keyids:
            bi = self.keysamplers[ki](pool) 
            #Tell other samplers that this sample was added
            #but only if these others won't be given NaN values
            if not self.nanOthers:
                for sm in self.keysamplers:
                    if sm == self.keysamplers[ki]:
                        continue
                    sm.add_other_sample(bi,pool)
            allbatchids.append(bi)
        return allbatchids,keyids 

    def _extractInds(self,pool,bId,kId):
        batch = {}
        for key in pool:
            batch[key] = pool[key][bId,...]
            if self.nanOthers and key in self.keylist:
                nanids = [k for k,v in enumerate(kId) if self.keylist[v]!=key]
                batch[key][nanids,...] = np.nan
        return batch

    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k == 'POSSIBLE_METHODS':
                continue
            if k == 'keysamplers':
                properties[k] = [i.to_dict() for i in self.__dict__[k] if hasattr(i,'to_dict')]
            elif k in {'pooler'}:
                properties[k] = self.__dict__[k].to_dict()
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties


class CategoricalSampler(object):
    def __init__(self,labelKey,pickLowestFrequency=False,countOthers=False):
        self.labelKey = labelKey
        self.pickLowestFrequency = pickLowestFrequency
        self.massSoFar = None
        self.idsSoFar = []
        self.countOthers = countOthers

    def __call__(self,data):
        labels = data[self.labelKey] 
        if self.massSoFar is None:
            self.massSoFar = np.zeros((1,labels.shape[1])).astype(theano.config.floatX)
        if self.pickLowestFrequency:
            categoryToFind = np.argmin(self.massSoFar)
        else:
            categoryToFind= np.random.choice(np.arange(labels.shape[1]),size=1)

        labelsCT = labels[:,categoryToFind].flatten().astype(theano.config.floatX)
        labelindsTotal = np.nansum(labelsCT)
        if labelindsTotal == 0:
            exampleProbs = np.ones_like(labelsCT,dtype=float)
            exampleProbs[np.isnan(labelsCT)] = 0
            exampleProbs /= np.sum(exampleProbs)
        else:
            exampleProbs = labelsCT/np.nansum(labelsCT)
            exampleProbs[np.isnan(exampleProbs)]=0
        ep = exampleProbs.astype(float)/np.sum(exampleProbs.astype(float))
        exampleInd = np.random.choice(np.arange(labelsCT.shape[0]),p=ep,size=1)
        self._add_one_sample(exampleInd[0],labels)
        return exampleInd[0]

    def _add_one_sample(self,sampleid,labels):
        if self.massSoFar is None:
            self.massSoFar = np.zeros((1,labels.shape[1])).astype(theano.config.floatX)
        self.idsSoFar.append(sampleid)
        self.massSoFar += np.nansum(labels[[self.idsSoFar[-1]],:],axis=0,keepdims=True)

    def add_other_sample(self,sampleid,data):
        if self.countOthers:
            self._add_one_sample(sampleid,data[self.labelKey])

    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k in {'idsSoFar'}:
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

class BinarySampler(object):
    def __init__(self,labelKey,countOthers=False):
        self.labelKey = labelKey
        self.idsSoFar = []
        self.numPos = 0
        self.numNeg = 0
        self.countOthers = countOthers

    def __call__(self,data):
        labels = data[self.labelKey]
        pickPosNeg = self.numPos < self.numNeg if self.numPos != self.numNeg else np.random.randint(0,2)
        try:
            exampleInd = np.random.choice(np.where(labels.flatten()==pickPosNeg)[0],size=1)
        except:
            exampleInd = np.random.choice(np.arange(labelsCT.shape[0]),size=1)
        self._add_one_sample(exampleInd[0],labels)
        return exampleInd[0]

    def _add_one_sample(self,sampleid,labels):
        self.idsSoFar.append(sampleid)
        self.numPos += np.nansum(labels[self.idsSoFar[-1],:])
        self.numNeg += np.nansum(1-labels[self.idsSoFar[-1],:])

    def add_other_sample(self,sampleid,data):
        if self.countOthers:
            self._add_one_sample(sampleid,data[self.labelKey])
    
    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k in {'idsSoFar'}:
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties

class BinnedSampler(object):
    def __init__(self,labelKey,bins,countOthers=False):
        self.labelKey = labelKey
        self.bins=bins
        self.bincounts = np.zeros((len(self.bins),))
        self.idsSoFar = []
        self.countOthers = countOthers

    def __call__(self,data):
        labels = data[self.labelKey]
        labelsBinned = np.digitize(labels.flatten(),bins=self.bins)
        binToFind = np.argmin(self.bincounts)
        try:
            exampleInd = np.random.choice(np.where(labelsBinned.flatten()==binToFind)[0],size=1)
        except:
            exampleInd = np.random.choice(np.arange(labels.shape[0]),size=1)
        self._add_one_sample(exampleInd[0],labelsBinned)
        return exampleInd[0]

    def _add_one_sample(self,sampleid,labelsBinned):
        self.idsSoFar.append(sampleid)
        self.bincounts += np.bincount([labelsBinned[self.idsSoFar[-1]]],minlength=len(self.bins))[0:len(self.bins)]

    def add_other_sample(self,sampleid,data):
        if self.countOthers:
            self._add_one_sample(sampleid,np.digitize(data[self.labelKey].flatten(),bins=self.bins))

    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k in {'idsSoFar'}:
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties


