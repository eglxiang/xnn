import numpy as np
from copy import deepcopy
import theano

class Sampler(object):
    def __init__(self, pooler,keysamplers=[],samplemethod='sequential',batchsize=128, numbatches=None, nanOthers=False):
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
            self._reset_batch()
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
        
    def _reset_batch(self):
        for ks in self.keysamplers:
            ks.reset_batch()

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
        self.newbatch = True
        self.exampleProbs = None
        self.labels = None
        self.exshape = None

    def __call__(self,data):

        if self.newbatch:
            self._set_up_new_batch(data)

        if self.pickLowestFrequency:
            categoryToFind = np.argmin(self.massSoFar)
        else:
            categoryToFind= np.random.choice(np.arange(self.ncat),size=1)[0]

        exampleInd = np.random.choice(self.exshape[categoryToFind],p=self.exampleProbs[categoryToFind],size=1)
        self._add_one_sample(exampleInd[0],self.labels)
        return exampleInd[0]

    def _set_up_new_batch(self,data):
        labels = data[self.labelKey]
        self.labels = labels
        self.ncat = labels.shape[1]
        self.nex = labels.shape[0]
        self.massSoFar = np.zeros((1,labels.shape[1])).astype(theano.config.floatX)
        self.exampleProbs = {}
        self.exshape = {}
        for i in xrange(labels.shape[1]):
            labelsCT = labels[:,i].flatten().astype(theano.config.floatX)
            labelindsTotal = np.nansum(labelsCT)
            if labelindsTotal == 0:
                exampleProbs = np.ones_like(labelsCT,dtype=float)
                exampleProbs[np.isnan(labelsCT)] = 0
                exampleProbs /= np.sum(exampleProbs)
            else:
                exampleProbs = labelsCT/np.nansum(labelsCT)
                exampleProbs[np.isnan(exampleProbs)]=0
            ep = exampleProbs.astype(float)/np.sum(exampleProbs.astype(float))
            self.exampleProbs[i] = ep
            self.exshape[i] = np.arange(self.exampleProbs[i].shape[0])
        self.newbatch=False

    def _add_one_sample(self,sampleid,labels):
        if self.massSoFar is None:
            self.massSoFar = np.zeros((1,labels.shape[1])).astype(theano.config.floatX)
        self.idsSoFar.append(sampleid)
        l = labels[sampleid,:]
        l[np.isnan(l)] = 0
        self.massSoFar += l 

    def reset_batch(self):
        self.newbatch=True

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
        self.posnegIds = None
        self.labels = None
        self.newbatch = True
        self.countOthers = countOthers

    def __call__(self,data):
        if self.newbatch:
            self._set_up_new_batch(data)

        pickPosNeg = self.numPos < self.numNeg if self.numPos != self.numNeg else np.random.randint(0,2)
        try:
            exampleInd = np.random.choice(self.posnegIds[pickPosNeg],size=1)
        except:
            exampleInd = np.random.choice(np.arange(self.labels.shape[0]),size=1)
        self._add_one_sample(exampleInd[0],self.labels)
        return exampleInd[0]
            
    
    
    def _set_up_new_batch(self,data):
        self.labels = data[self.labelKey]
        self.posnegIds = []
        self.posnegIds.append(np.where(self.labels.flatten()==0)[0])
        self.posnegIds.append(np.where(self.labels.flatten()==1)[0])
        self.newbatch=False

    def _add_one_sample(self,sampleid,labels):
        self.idsSoFar.append(sampleid)
        l = labels[sampleid,0]
        if l!=l:
            return
        self.numPos += l 
        self.numNeg += 1-l 

    def reset_batch(self):
        self.newbatch = True

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
        self.labelsBinned = None
        self.binids = None
        self.nexrange = None

    def __call__(self,data):

        if self.newbatch:
            self._set_up_new_batch(data)

        binToFind = np.argmin(self.bincounts)
        try:
            exampleInd = np.random.choice(self.binids[binToFind],size=1)
        except:
            exampleInd = np.random.choice(self.nexrange,size=1)
        self._add_one_sample(exampleInd[0],self.labelsBinned)
        return exampleInd[0]

    def _set_up_new_batch(self,data):
        labels = data[self.labelKey]
        self.nexrange = np.arange(labels.shape[0])
        self.labelsBinned = np.digitize(labels.flatten(),bins=self.bins)
        self.binids = {}
        for i in xrange(len(self.bincounts)):
            self.binids[i] = np.where(self.labelsBinned.flatten()==i)[0]
        self.newbatch=False


    def _add_one_sample(self,sampleid,labelsBinned):
        self.idsSoFar.append(sampleid)
        l = labelsBinned[sampleid]
        if l >= len(self.bincounts):
            return
        self.bincounts[l] += 1

    def add_other_sample(self,sampleid,data):
        if self.countOthers:
            self._add_one_sample(sampleid,np.digitize(data[self.labelKey].flatten(),bins=self.bins))
    
    def reset_batch(self):
        self.newbatch=True

    def to_dict(self):
        properties = {}
        for k in self.__dict__:
            if k in {'idsSoFar'}:
                continue
            else:
                properties[k] = deepcopy(self.__dict__[k])
        return properties


