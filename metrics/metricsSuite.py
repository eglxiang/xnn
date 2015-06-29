"""
A collection of metrics for evaluating detectors written by Jake Whitehill
and adapted by Josh Susskind
"""

# In this file, x is the detector's output, and y is the true object label
import pylab
import numpy as np
import math
import sys

__all__=['metric_types','metric_names','Metric',
            'baseline',
            'confMatAggregate',
            'compute2AFC',
            'computeBalancedErrorRate',
            'computeBalancedExponentialCost',
            'computeBalancedLogisticAndExponentialCosts',
            'computeBalancedLogisticCost',
            'computeBinarizedBalancedErrorRateBinary',
            'computeBinarizedBalancedErrorRateCategorical',
            'computeBinarizedBalancedExponentialCost',
            'computeBinarizedBalancedLogisticCost',
            'computeBinarizedF1',
            'computeBinarizedHitRate',
            'computeBinarizedJunkRate',
            'computeBinarizedSpecificity',
            'computeCategoricalCrossentropy',
            'computeConfusionMatrix',
            'computeCondProb',
            'computeEqualErrorRate',
            'computeErrorRateDiffSquared',
            'computeF',
            'computeF1',
            'computeHitRate',
            'computeJCorr',
            'computeJunkRate',
            'computeKLDivergence',
            'computeOptimalBalancedErrorRate',
            'computeOptimalBalancedExponentialCost',
            'computeOptimalBalancedLogisticCost',
            'computeOptimalF1',
            'computePercentCorrect',
            'computePrecision',
            'computeSpecificity',
            'computeThresholdPercentCorrect',
            'convertLLRtoProb',
            'optimizeOverBaselinesAndScales',
            'optimizeOverThresholds',
            'threshold',
            ]


metric_types={}
metric_names={}

class Metric():
    def __init__(self,metric,targkeys,outkeys=None,weightkey=None,aggregation_type='mean',**kwargs):
        if type(metric) == str:
            self.metric = metric_types[metric.lower()]
        else:
            assert hasattr(metric,'__call__')
            self.metric = metric
        if self.metric in metric_names:
            self.name=metric_names[self.metric]
        else:
            self.name=self.metric.__name__

        self.settings  = kwargs
        self.weightkey = weightkey
        self.aggregation_type=aggregation_type

        if not (isinstance(targkeys,str) or isinstance(targkeys,list)):
            raise TypeError("targetkeys needs to be a string or a list of strings")
        self.targkeys = targkeys
        if outkeys is not None and not isinstance(outkeys,list):
            outkeys = [outkeys]
        self.outkeys  = outkeys

    def __call__(self,out,datadict):

        # if outkeys is a list, pass the value of that list extracted from the output
        if self.outkeys is not None and isinstance(out,dict):
            olist = [out[ok] for ok in self.outkeys]
            if len(olist==1):
                out = olist[0]
            else:
                out = olist 
            
        if isinstance(self.targkeys,str):
            targ = datadict[self.targkeys]
        else:
            targ = dict()
            for k in self.targkeys:
                targ[k] = datadict[k]
        settings_copy = self.settings.copy()
        output        = self.metric(out,targ,**settings_copy)
        weights       = datadict[self.weightkey] if self.weightkey is not None else None
        if self.aggregation_type == 'mean':
            output = np.nanmean(output)
        elif self.aggregation_type == 'sum':
            output = np.nansum(output)
        elif self.aggregation_type == 'weighted_mean':
            if weights is None:
                raise Exception('weighted_mean aggregation requires specifying a weightkey')
            output = np.nansum(output*weights)/np.nansum(weights)
        elif self.aggregation_type == 'weighted_sum':
            if weights is None:
                raise Exception('weighted_sum aggregation requires specifying a weightkey')
            output = np.nansum(output*weights)
        elif hasattr(self.aggregation_type,'__call__'):
            if weights is None:
                output = self.aggregation_type(output)
            else:
                output = self.aggregation_type(output,weights)
        else:
            pass
        return output

    def to_dict(self):
        d = self.__dict__.copy() 
        d['metric'] = d['metric'].__name__
        if hasattr(d['aggregation_type'],'__call__'):
            d['aggregation_type'] = d['aggregation_type'].__name__
        return d

def computeKLDivergence(x,y):
    x      = np.float32(x)
    y      = np.float32(y)
    lograt = np.log(y) - np.log(x)
    lograt[np.isinf(lograt)] = 50
    return np.nansum(y*lograt,axis=1)
metric_types['kl']=computeKLDivergence 
metric_names[computeKLDivergence]='KL Divergence'

def computeSquaredError(x,y):
    return x**2-y**2
metric_types['se']=computeSquaredError 
metric_names[computeSquaredError]='Squared Error'

def computeAbsoluteError(x,y):
    return np.abs(x-y)
metric_types['ae']=computeAbsoluteError 
metric_names[computeSquaredError]='Absolute Error'

def computeCategoricalCrossentropy(x,y):
    x = np.float32(x)
    y = np.float32(y)
    logpred = np.log(x)
    # deal with predictions that are 0
    logpred[np.isinf(logpred)]=-50
    xlp = y*logpred
    if xlp.ndim == 1:
        xlp = xlp[:,None]
    return -np.sum(xlp,axis=1)
metric_types['categorical_crossentropy']=computeCategoricalCrossentropy
metric_types['cce']=computeCategoricalCrossentropy
metric_names[computeCategoricalCrossentropy]='Categorical Crossentropy'

def computeConfusionMatrix(x,y):
    ym = np.argmax(y,axis=1)
    xm = np.argmax(x,axis=1)
    n  = y.shape[1]+1
    ym[np.any(np.isnan(y),axis=1)] = n-1
    ntp       = n*ym + xm
    bc        = np.bincount(ntp,minlength=n*n).reshape((n,n))
    countsmat = bc[:-1,:-1]
    return countsmat

def baseline (x, b = 0):
    return np.array(x, float) - b

def threshold (x, t = 0):
    return np.array(np.array(x, float) > t, float)

def computeCondProb (a, b, l):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if l == 0:
        a = 1-a
        b = 1-b
    c = a*b
    d = np.sum(c,axis=0)/np.sum(b,axis=0)
    return d 
# metric_types['condprob']=computeCondProb
# metric_types['cp']=computeCondProb
# metric_names[computeCondProb]='Conditional probability'

# Compute P(Y=0 | X=1, Tag=1) = 1 - precision
def computeJunkRate (x, y):
    precision = computePrecision(x, y)
    if precision is None:
        return None
    return 1 - precision
metric_types['junkrate']=computeJunkRate
metric_types['junk rate']=computeJunkRate
metric_types['jr']=computeJunkRate
metric_names[computeJunkRate]='Junk Rate'

def computeBinarizedHitRate (x, y, t=0):
    x = threshold(x,t)
    return computeHitRate(x, y)
metric_types['binarizedhitrate']=computeBinarizedHitRate
metric_types['bhr']=computeBinarizedHitRate
metric_names[computeBinarizedHitRate]='Binarized Hit Rate'

def computeBinarizedSpecificity (x, y, t=0):
    x = threshold(x,t)
    return computeSpecificity(x, y)
metric_types['binarizedspecificity']=computeBinarizedSpecificity
metric_types['bs']=computeBinarizedSpecificity
metric_names[computeBinarizedSpecificity]='Binarized Specificity'

def computeBinarizedJunkRate (x, y, t=0):
    x = threshold(x,t)
    return computeJunkRate(x, y)
metric_types['binarizedjunkrate']=computeBinarizedJunkRate
metric_types['binarizedjr']=computeBinarizedJunkRate
metric_types['bjr']=computeBinarizedJunkRate
metric_names[computeBinarizedJunkRate]='Binarized Junk Rate'

# Compute P(Y=1 | X=1, Tag=1)
def computePrecision (x, y):
    return computeCondProb(y, x, 1)
metric_types['precision']=computePrecision
metric_names[computePrecision]='Precision'
# Compute P(X=0 | Y=0, Tag=1)
def computeSpecificity (x, y):
    return computeCondProb(x, y, 0)
metric_types['specificity']=computeSpecificity
metric_names[computeSpecificity]='Specificity'

# Compute P(X=1 | Y=1, Tag=1)
def computeHitRate (x, y):
    return computeCondProb(x, y, 1)
metric_types['hitrate']=computeHitRate
metric_types['hit rate']=computeHitRate
metric_types['hr']=computeHitRate
metric_names[computeHitRate]='Hit Rate'

def computeOptimalBalancedErrorRate (x, y, numThresholds = 100):
    return optimizeOverThresholds(computeBalancedErrorRate,
                                    False, x, y,
                                    numThresholds)
metric_types['optimalbalancederrorrate']=computeOptimalBalancedErrorRate
metric_types['optimalbalancederror']=computeOptimalBalancedErrorRate
metric_types['ober']=computeOptimalBalancedErrorRate
metric_types['obe']=computeOptimalBalancedErrorRate
metric_names[computeOptimalBalancedErrorRate]='Optimal Balanced Error Rate'

def computeBinarizedBalancedErrorRateBinary(x, y, t=0.5):
    x = threshold(x,t)
    y = threshold(y,t)
    return computeBalancedErrorRate(x, y)
metric_types['binarizedbalancederrorratebinary']=computeBinarizedBalancedErrorRateBinary
metric_types['binarizedbalancederrorbinary']=computeBinarizedBalancedErrorRateBinary
metric_types['bberb']=computeBinarizedBalancedErrorRateBinary
metric_types['bbeb']=computeBinarizedBalancedErrorRateBinary
metric_names[computeBinarizedBalancedErrorRateBinary]='Binarized Balanced Error Rate Binary'

def _softmax(w, t = 1.0):
    e = np.exp(np.float32(w) / t)
    dist = e / np.sum(e,axis=1,keepdims=True)
    return dist

def _convertToBinaryProb(p):
    num_other_classes = p.shape[1] - 1
    q = (1-p)/num_other_classes
    q[q==0]=1
    pc = _softmax(np.log(p)-np.log(q))
    return pc

def computeBinarizedBalancedErrorRateCategorical(x,y,t=0.5):
    return computeBinarizedBalancedErrorRateBinary(_convertToBinaryProb(x),_convertToBinaryProb(y),t)
metric_types['binarizedbalancederrorratecategorical']=computeBinarizedBalancedErrorRateCategorical
metric_types['binarizedbalancederrorcategorical']=computeBinarizedBalancedErrorRateCategorical
metric_types['bberc']=computeBinarizedBalancedErrorRateCategorical
metric_types['bbec']=computeBinarizedBalancedErrorRateCategorical
metric_names[computeBinarizedBalancedErrorRateCategorical]='Binarized Balanced Error Rate Categorical'

def computeBalancedErrorRate (x, y):
    r = computeHitRate(x, y)
    s = computeSpecificity(x, y)
    r[np.isnan(r)] = 0
    s[np.isnan(s)] = 0
    if (r is None) or (s is None):
        return None
    return 0.5 * (1 - r) + 0.5 * (1 - s)
metric_types['balancederrorrate']=computeBalancedErrorRate
metric_types['balancederror']=computeBalancedErrorRate
metric_types['ber']=computeBalancedErrorRate
metric_types['be']=computeBalancedErrorRate
metric_names[computeBalancedErrorRate]='Balanced Error Rate'

def computeF (x, y, alpha, beta):
    p = computePrecision(x, y)
    r = computeHitRate(x, y)
    if (p is None) or (r is None): #or (p == 0) or (r == 0):
        return - float("inf")
    return (alpha / p + (1 - alpha) / r) ** -1
metric_types['f']=computeF
metric_names[computeF]='F'

def computeErrorRateDiffSquared (x, y):
    hitRate = computeHitRate(x, y)
    if hitRate is None:
        return None
    missRate = 1 - hitRate
    falseAlarmRate = 1 - computeSpecificity(x, y)
    diffSquared = (missRate - falseAlarmRate) ** 2
    return diffSquared
metric_types['errorratediffsquared']=computeErrorRateDiffSquared
metric_types['erds']=computeErrorRateDiffSquared
metric_names[computeErrorRateDiffSquared]='Error Rate Difference Squared'

def computePercentCorrect (x, y):
    return np.sum(np.array(x == y, float)) / len(x)
metric_types['percentcorrect']=computePercentCorrect
metric_types['percent correct']=computePercentCorrect
metric_types['pc']=computePercentCorrect
metric_names[computePercentCorrect]='Percent Correct'

def computeThresholdPercentCorrect (x, y, t=0.5):
    x=threshold(x,t)
    return np.sum(np.array(x == y, float)) / len(x)
metric_types['thresholdpercentcorrect']=computeThresholdPercentCorrect
metric_types['threshold percent correct']=computeThresholdPercentCorrect
metric_types['tpc']=computeThresholdPercentCorrect
metric_names[computeThresholdPercentCorrect]='Threshold Percent Correct'

def computeEqualErrorRate (x, y):
    # Find threshold that minimizes squared difference between FP and FN rates
    (best,bestT) = optimizeOverThresholds(computeErrorRateDiffSquared, False, x, y)
    # For that threshold, return the percent correct
    x = threshold(x, bestT)
    return computePercentCorrect(x, y)
metric_types['equalerrorrate']=computeEqualErrorRate
metric_types['equalerror']=computeEqualErrorRate
metric_types['equal errorrate']=computeEqualErrorRate
metric_types['equal error']=computeEqualErrorRate
metric_types['eer']=computeEqualErrorRate
metric_types['ee']=computeEqualErrorRate
metric_names[computeEqualErrorRate]='Equal Error Rate'

def optimizeOverBaselinesAndScales (metricFn, shouldMaximize, x, y):
    # Compute array of all useful thresholds (x - eps for every x, and also max(x) + eps)
    x = np.array(x)
    minBaseline = min(x)
    maxBaseline = max(x)
    numBaselines = 100

    minScale = 1e-1
    maxScale = 1e+1
    numScales = 10
    scaleFactor = (maxScale/minScale) ** (1./numScales)
    if shouldMaximize:
        best = -float("inf")
        bestB = 0
        bestS = 0
    else:
        best = +float("inf")
        bestB = 0
        bestS = 0
    origX = x
    for i in range(numBaselines):
        b = minBaseline + float(i) * (maxBaseline - minBaseline) / numBaselines
        for j in range(numScales):
            scale = minScale * scaleFactor ** j
            x = baseline(origX * scale, b)
            value = metricFn(x, y)
            if shouldMaximize and (value > best):
                best = value
                bestB = b
                bestS = scale
            elif not shouldMaximize and (value < best):
                best = value
                bestB = b
                bestS = scale
    if (best == float("inf")) or (best == - float("inf")):
        # If we never found a valid value, then return None
        best = None
    return (best, bestB, bestS)

def optimizeOverThresholds (metricFn, shouldMaximize, x, y,numThresholds = None):
    # Compute array of all useful thresholds
    # (x - eps for every x, and also max(x) + eps)
    x = np.array(x)
    eps = sys.float_info.epsilon

    if numThresholds is not None:
        thresholds = np.arange(np.min(x), np.max(x), (np.max(x)-np.min(x))/float(numThresholds))
    else:
        thresholds = x - eps
        thresholds = np.append(thresholds, np.max(x) + eps)

    if shouldMaximize:
        best = -float("inf")
        bestT = 0
    else:
        best = +float("inf")
        bestT = 0
    origX = x
    for t in thresholds:
        x = threshold(origX, t)
        value = np.nanmean(metricFn(x, y))
        if shouldMaximize and (value > best):
            best = value
            bestT = t
        elif not shouldMaximize and (value < best):
            best = value
            bestT = t
    if (best == float("inf")) or (best == - float("inf")):
        # If we never found a valid value, then return None
        best = None
    return (best, bestT)

def computeOptimalF1 (x, y, numThresholds = 100):
    return optimizeOverThresholds(computeF1, True, x, y, numThresholds)
metric_types['optimalf1']=computeOptimalF1
metric_types['of1']=computeOptimalF1
metric_names[computeOptimalF1]='Optimal F1'

def computeBinarizedF1 (x, y, t=0):
    x = threshold(x,t)
    return computeF1(x, y)
metric_types['binarizedf1']=computeBinarizedF1
metric_types['bf1']=computeBinarizedF1
metric_names[computeBinarizedF1]='Binarized F1'

def computeF1 (x, y):
    return computeF(x, y, 0.5, 1)
metric_types['f1']=computeF1
metric_names[computeF1]='F1'

# Convert a log-likelihood ratio into a probability (around baseline b)
def convertLLRtoProb (x, b):
    return 1. / (1. + math.exp(- (x - b)))

def computeOptimalBalancedLogisticCost (x, y):
    return optimizeOverBaselinesAndScales(computeBalancedLogisticCost, False, x, y)
metric_types['optimalbalancedlogisticcost']=computeOptimalBalancedLogisticCost
metric_types['oblc']=computeOptimalBalancedLogisticCost
metric_names[computeOptimalBalancedLogisticCost]='Optimal Balanced Logistic Cost'

def computeOptimalBalancedExponentialCost (x, y):
    return optimizeOverBaselinesAndScales(computeBalancedExponentialCost, False, x, y)
metric_types['optimalbalancedexponentialcost']=computeOptimalBalancedExponentialCost
metric_types['obec']=computeOptimalBalancedExponentialCost
metric_names[computeOptimalBalancedExponentialCost]='Optimal Balanced Exponential Cost'

def computeBinarizedBalancedLogisticCost (x, y):
    return computeBalancedLogisticCost(x, y)
metric_types['binarizedbalancedlogisticcost']=computeBinarizedBalancedLogisticCost
metric_types['bblc']=computeBinarizedBalancedLogisticCost
metric_names[computeBinarizedBalancedLogisticCost]='Binarized Balanced Logistic Cost'

def computeBinarizedBalancedExponentialCost (x, y):
    return computeBalancedExponentialCost(x, y)
metric_types['binarizedbalancedexponentialcost']=computeBinarizedBalancedExponentialCost
metric_names[computeBinarizedBalancedExponentialCost]='Binarized Balanced Exponential Cost'

def computeBalancedLogisticCost (x, y):
    return computeBalancedLogisticAndExponentialCosts(x, y)[0]
metric_types['balancedlogisticcost']=computeBalancedLogisticCost
metric_types['blc']=computeBalancedLogisticCost
metric_names[computeBalancedLogisticCost]='Balanced Logistic Cost'

def computeBalancedExponentialCost (x, y):
    return computeBalancedLogisticAndExponentialCosts(x, y)[1]
metric_types['balancedexponentialcost']=computeBalancedExponentialCost
metric_types['bec']=computeBalancedExponentialCost
metric_names[computeBalancedExponentialCost]='Balanced Exponential Cost'

def computeBalancedLogisticAndExponentialCosts (x, y):
    x = np.array(x, float)
    y = np.array(y, float)
    # If specified, limit ourselves to data for which tag == 1
    idxs1 = pylab.find(y == 1)
    idxs0 = pylab.find(y == 0)
    n1 = len(idxs1)
    n0 = len(idxs0)
    logisticCost = 0
    exponentialCost = 0
    for idx in idxs1:
        p = convertLLRtoProb(x[idx], 0)
        if p*(1-p)==0:  # Precision error -- return "inf" as infinite cost so that, in optimization, we don't use the parameters that got us here
            return (float("inf"), float("inf"))
        logisticCost -= 0.5/n1 * math.log(p)
        exponentialCost += 0.5/n1 * abs(1 - p) / math.sqrt(p*(1-p))
    for idx in idxs0:
        p = convertLLRtoProb(x[idx], 0)
        if p*(1-p)==0:  # Precision error -- return "inf" as infinite cost so that, in optimization, we don't use the parameters that got us here
            return (float("inf"), float("inf"))
        logisticCost -= 0.5/n0 * math.log(1 - p)
        exponentialCost += 0.5/n0 * abs(0 - p) / math.sqrt(p*(1-p))
    return (logisticCost, exponentialCost)
metric_types['balancedlogisticandexponentialcosts']=computeBalancedLogisticAndExponentialCosts
metric_types['blec']=computeBalancedLogisticAndExponentialCosts
metric_types['blaec']=computeBalancedLogisticAndExponentialCosts
metric_names[computeBalancedLogisticAndExponentialCosts]='Balanced Logistic And Exponential Costs'

# Compute the 2AFC of responses x given ground-truth labels y.
# tags is an optional parameter and can be used to filter the
# set of pairs to make sure that at least one item in each pair
# has tag == 1.
def compute2AFC (x, y):
    x = np.array(x, float)
    y = np.array(y, float)

    idxs0 = pylab.find(y == c[0])
    idxs1 = pylab.find(y == c[1])
    x0 = x[idxs0]
    x1 = x[idxs1]

    n0 = len(x0)
    n1 = len(x1)

    c0 = np.zeros(n0, float)
    numPairs = np.zeros(n0, float)
    for k in range(0, n0):
        # If user passed an array of tags, only consider pairs in which at least one
        # element has tag == 1.
        valid = np.ones(n1, bool)
        c0[k] = len(pylab.find(x1[valid] > x0[k])) + 0.5 * len(pylab.find(x1[valid] == x0[k]))

    return np.sum(c0) / np.sum(numPairs)
metric_types['2afc']=compute2AFC
metric_names[compute2AFC]='2AFC'

def computeJCorr (clipsData):
    allX = np.array([])
    varWithin = 0
    for clip in clipsData:
        x = np.array(clip)
        varWithin += np.sum((x - np.mean(x)) ** 2)

        # Assemble all the x values into allX
        allX = np.append(allX, x)
    varTotal = np.sum((allX - np.mean(allX)) ** 2)
    return np.sqrt(1 - varWithin/varTotal)
# metric_types['jcorr']=computeJCorr


def confMatAggregate(c):
    cm = c.copy()
    c = c.astype(np.float32) / np.sum(c,axis=1)
    c[np.isnan(c)]=0
    return cm,np.nanmean(np.diag(c))

