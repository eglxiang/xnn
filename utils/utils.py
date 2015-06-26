import numpy as np
import theano 
import theano.tensor as T
import time
import sys
import pylab as pl
import signal

def theano_digitize(x, bins):
    """
    Equivalent to numpy digitize
    """
    binned = T.zeros_like(x)
    for i in range(len(bins)):
        bin=bins[i]
        if i == 0:
            binned=T.switch(T.lt(x,bin),i,binned)
        else:
            ineq = T.and_(T.ge(x,bins[i-1]),T.lt(x,bin))
            binned=T.switch(ineq,i,binned)
    binned=T.switch(T.isnan(x), len(bins), binned)
    return binned

def numpy_one_hot(x,numclasses=None):
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype=theano.config.floatX)
    z = np.zeros(x.shape, dtype=theano.config.floatX)
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z
    return result

def Tnanmean(x,axis=None,keepdims=False):
    mask1 = T.switch(T.isnan(x),0,x)
    mask2 = T.switch(T.isnan(x),0,1)
    return T.true_div(mask1.sum(axis=axis,keepdims=keepdims),mask2.sum(axis=axis,keepdims=keepdims))

def Tnanmax(x,axis=None,keepdims=False):
    mask = T.switch(T.isnan(x),-np.inf,x)
    return mask.max(axis=axis,keepdims=keepdims)

def Tnansum(x, axis=None, keepdims=False):
    mask = T.switch(T.isnan(x),0,x)
    return mask.sum(axis=axis,keepdims=keepdims)

class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v, 1]
                self.unique_values.append(k)
            else:
                if type(v)==str:
                    self.sum_values[k][0] = v
                else:
                    self.sum_values[k][0] += v * (current-self.seen_so_far)
                    self.sum_values[k][1] += (current-self.seen_so_far)

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * (self.total_width+1))
            sys.stdout.write("\r")

            bar = '%d/%d [' % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(k[0]) == str:
                    info += ' - %s: %s' % (k, self.sum_values[k][0])
                else:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0]/self.sum_values[k][1])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()
            self.seen_so_far = current

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0]/self.sum_values[k][1])
                sys.stdout.write(info + "\n")


    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)

class GracefulStop(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        self.count=0
        signal.signal(signal.SIGINT,self.handler)

    def handler(self,signal,frame):
        self.count+=1
        self.signal_received = (signal,frame)
        if self.count == 1:
            print "---Cancelling training after this epoch. To cancel immediately, press ctrl-C again---"
        else:
            signal.signal(signal.SIGINT,self.old_handler)

    def __exit__(self,type,value,traceback):
        signal.signal(signal.SIGINT,self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

class lengthExpection(Exception):
    pass

class nonNegativeExpection(Exception):
    pass

class noProbVectorException(Exception):
    pass


def probEmbarrasingMistakeForAge(fHuman,pMachine,bins=[18,25,35,45,55,65,100]):
# compute probabiliity of embarrasing mistake for age estimation
# fHuman is a n-by-100 dimensional vector of human estimates of age
# fHuman[j][k]=10 means that 10 people thought the age of person j was k
# pMachine is probability of machine producing an age estimate within a specific interval
# pMachine[j][0]=0.1  means that 10% of time the machine estimate the age of the person j is in category 0
# the  upper limits in years for each of the categories are in an array defined in bins

    categoryUpperLims=bins
    nc=len(categoryUpperLims)
    if type(fHuman)==list:
        fHuman = array(fHuman)
    if type(pMachine)==list:
        pMachine = array(pMachine)
    if fHuman.shape[1] != 100:
        raise lengthExpection('vector of human age estimates is not of dimension 100')
    if pMachine.shape[1]!=nc:
        raise lengthExpection('vector of probability estimates is not of dimension expected:', nc)
    if sum(fHuman<0)>0:
        raise nonNegativeExpection('vector of human age estimates has negative values')
    if sum(pMachine<0)>0:
        raise nonNegativeExpection('vector of machine probabilities contains negative values')
    if pMachine.sum()!=pMachine.shape[0]:
        raise nonNegativeExpection('vector of machine probabilities does not add up to one')

    # use frequency of human labels to do probabiity density estimation
    # with Gaussian kernels

    pHuman = zeros(fHuman.shape)
    for k1 in range(100):
        for k2 in range(100):
            d2= (k2-k1)**2.
            sd = 0.2*(k2+1.)  # slack in the age estimate is 20 % of the estimate
            s2 = sd**2.
            pHuman[:,k1] = pHuman[:,k1] + exp(-0.5*d2/s2) * fHuman[:,k2]/(sd+0.)

    pHuman = pHuman/pHuman.sum(1,keepdims=True)
# based on the year by year probabiilty estimates
# compute the probability of each age category

    pHumanC=zeros(pMachine.shape)
    c1=categoryUpperLims[0]+1
    c0=0
    c1=categoryUpperLims[0]
    pHumanC[:,0]=pHuman[:,0:c1+1].sum(1)
    for k in range(1,nc):
        c0=categoryUpperLims[k-1]+1
        c1=categoryUpperLims[k]
        pHumanC[:,k]=pHuman[:,c0:c1+1].sum(1)


# a machine probability estimate is not embarrasing if either one of the two people agrees with us, or
# the two people provide estimates that bracket our estimate. If X1 and X2 are the two human estimates
# and y is the machine estimate then
# prob(no embarrasing | y)= p(X1 <= y)p(X2 >= y)

# Compute the expected probability of embarrasing (1 - prob OK)
    probOK=zeros((fHuman.shape[0],))
    for y in range(nc):
    #y is the index to the category chosen by humans
        Fy = pHumanC[:,0:y+1].sum(1)
        Gy = (1.-Fy)+pHumanC[:,y]
        probOK = probOK+pMachine[:,y]*Fy*Gy
    return 1.-probOK


def typechecker(f):
    def typecheck(*args,**kwargs):
        all_args = list(args)+kwargs.values()
        tmp = T.matrix()
        for a in all_args:
            if type(a)==type(tmp):
                return f(*args,**kwargs)
        return f(*args,**kwargs).eval()
    return typecheck