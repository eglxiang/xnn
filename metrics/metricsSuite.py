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
            'compute2AFC',
            'computeBalancedErrorRate',
            'computeBalancedExponentialCost',
            'computeBalancedLogisticAndExponentialCosts',
            'computeBalancedLogisticCost',
            'computeBinarizedBalancedErrorRate',
            'computeBinarizedBalancedExponentialCost',
            'computeBinarizedBalancedLogisticCost',
            'computeBinarizedF1',
            'computeBinarizedHitRate',
            'computeBinarizedJunkRate',
            'computeBinarizedSpecificity',
            'computeCondProb',
            'computeEqualErrorRate',
            'computeErrorRateDiffSquared',
            'computeF',
            'computeF1',
            'computeHitRate',
            'computeJCorr',
            'computeJunkRate',
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
    def __init__(self,metric,targkeys,**kwargs):
        if type(metric) == str:
            self.metric = metric_types[metric.lower()]
        else:
            assert hasattr(metric,'__call__')
            self.metric = metric
        if self.metric in metric_names:
            self.name=metric_names[self.metric]
        else:
            self.name=self.metric.__name__
        self.settings = kwargs

        if not (isinstance(targkeys,str) or isinstance(targkeys,list)):
            raise TypeError("targetkeys needs to be a string or a list of strings")
        self.targkeys = targkeys

    def __call__(self,out,datadict):

        settings_copy=self.settings.copy()
        if isinstance(self.targkeys,str):
            targ = datadict[self.targkeys]
            output = self.metric(out,targ,**settings_copy)
        else:
            targ = dict()
            for k in self.targkeys:
                targ[k] = datadict[k]
            output = self.metric(out,targ,**settings_copy)
        return output

    def to_dict(self):
        d = self.__dict__.copy() 
        d['metric'] = d['metric'].__name__
        return d
                 
        

def baseline (x, b = 0):
    return np.array(x, float) - b

def threshold (x, t = 0):
    return np.array(np.array(x, float) > t, float)

# If tags is empty, then compute P(A=l | B=l);
# if tags is not empty, then compute P(A=l | B=l, Tag=1)


def computeCondProb (a, b, l, tags=None):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    # If specified, limit ourselves to data for which tag == 1
    if tags is not None:
        tags = np.array(tags, float)
        a = a[pylab.find(tags == 1)]
        b = b[pylab.find(tags == 1)]
    idxs = pylab.find(b == l)
    b = b[idxs]
    a = a[idxs]
    if len(b) == 0:
        return None
    return np.sum(np.array(a == b, float)) / len(b)
# metric_types['condprob']=computeCondProb
# metric_types['cp']=computeCondProb
# metric_names[computeCondProb]='Conditional probability'

# Compute P(Y=0 | X=1, Tag=1) = 1 - precision
def computeJunkRate (x, y, tags = None):
    precision = computePrecision(x, y, tags)
    if precision == None:
        return None
    return 1 - precision
metric_types['junkrate']=computeJunkRate
metric_types['junk rate']=computeJunkRate
metric_types['jr']=computeJunkRate
metric_names[computeJunkRate]='Junk Rate'

def computeBinarizedHitRate (x, y, tags = None, t=0):
    x = threshold(x,t)
    return computeHitRate(x, y, tags)
metric_types['binarizedhitrate']=computeBinarizedHitRate
metric_types['bhr']=computeBinarizedHitRate
metric_names[computeBinarizedHitRate]='Binarized Hit Rate'

def computeBinarizedSpecificity (x, y, tags = None, t=0):
    x = threshold(x,t)
    return computeSpecificity(x, y, tags)
metric_types['binarizedspecificity']=computeBinarizedSpecificity
metric_types['bs']=computeBinarizedSpecificity
metric_names[computeBinarizedSpecificity]='Binarized Specificity'

def computeBinarizedJunkRate (x, y, tags = None, t=0):
    x = threshold(x,t)
    return computeJunkRate(x, y, tags)
metric_types['binarizedjunkrate']=computeBinarizedJunkRate
metric_types['binarizedjr']=computeBinarizedJunkRate
metric_types['bjr']=computeBinarizedJunkRate
metric_names[computeBinarizedJunkRate]='Binarized Junk Rate'

# Compute P(Y=1 | X=1, Tag=1)
def computePrecision (x, y, tags = None):
    return computeCondProb(y, x, 1, tags)
metric_types['precision']=computePrecision
metric_names[computePrecision]='Precision'
# Compute P(X=0 | Y=0, Tag=1)
def computeSpecificity (x, y, tags = None):
    return computeCondProb(x, y, 0, tags)
metric_types['specificity']=computeSpecificity
metric_names[computeSpecificity]='Specificity'

# Compute P(X=1 | Y=1, Tag=1)
def computeHitRate (x, y, tags = None):
    return computeCondProb(x, y, 1, tags)
metric_types['hitrate']=computeHitRate
metric_types['hit rate']=computeHitRate
metric_types['hr']=computeHitRate
metric_names[computeHitRate]='Hit Rate'

def computeOptimalBalancedErrorRate (x, y, numThresholds = 100, tags = None):
    return optimizeOverThresholds(computeBalancedErrorRate,
                                    False, x, y,
                                    numThresholds, tags)
metric_types['optimalbalancederrorrate']=computeOptimalBalancedErrorRate
metric_types['optimalbalancederror']=computeOptimalBalancedErrorRate
metric_types['ober']=computeOptimalBalancedErrorRate
metric_types['obe']=computeOptimalBalancedErrorRate
metric_names[computeOptimalBalancedErrorRate]='Optimal Balanced Error Rate'

def computeBinarizedBalancedErrorRate (x, y, tags = None, t=0.5):
    x = threshold(x,t)
    return computeBalancedErrorRate(x, y, tags)
metric_types['binarizedbalancederrorrate']=computeBinarizedBalancedErrorRate
metric_types['binarizedbalancederror']=computeBinarizedBalancedErrorRate
metric_types['bber']=computeBinarizedBalancedErrorRate
metric_types['bbe']=computeBinarizedBalancedErrorRate
metric_names[computeBinarizedBalancedErrorRate]='Binarized Balanced Error Rate'

def computeBalancedErrorRate (x, y, tags = None):
    r = computeHitRate(x, y, tags)
    s = computeSpecificity(x, y, tags)
    if (r == None) or (s == None):
        return None
    return 0.5 * (1 - r) + 0.5 * (1 - s)
metric_types['balancederrorrate']=computeBalancedErrorRate
metric_types['balancederror']=computeBalancedErrorRate
metric_types['ber']=computeBalancedErrorRate
metric_types['be']=computeBalancedErrorRate
metric_names[computeBalancedErrorRate]='Balanced Error Rate'

def computeF (x, y, alpha, beta, tags=None):
    p = computePrecision(x, y, tags)
    r = computeHitRate(x, y, tags)
    if (p == None) or (r == None) or (p == 0) or (r == 0):
        return - float("inf")
    return (alpha / p + (1 - alpha) / r) ** -1
metric_types['f']=computeF
metric_names[computeF]='F'

def computeErrorRateDiffSquared (x, y, tags=None):
    hitRate = computeHitRate(x, y, tags)
    if hitRate == None:
        return None
    missRate = 1 - hitRate
    falseAlarmRate = 1 - computeSpecificity(x, y, tags)
    diffSquared = (missRate - falseAlarmRate) ** 2
    return diffSquared
metric_types['errorratediffsquared']=computeErrorRateDiffSquared
metric_types['erds']=computeErrorRateDiffSquared
metric_names[computeErrorRateDiffSquared]='Error Rate Difference Squared'

def computePercentCorrect (x, y, tags=None):
    return np.sum(np.array(x == y, float)) / len(x)
metric_types['percentcorrect']=computePercentCorrect
metric_types['percent correct']=computePercentCorrect
metric_types['pc']=computePercentCorrect
metric_names[computePercentCorrect]='Percent Correct'

def computeThresholdPercentCorrect (x, y, tags=None, t=0.5):
    x=threshold(x,t)
    return np.sum(np.array(x == y, float)) / len(x)
metric_types['thresholdpercentcorrect']=computeThresholdPercentCorrect
metric_types['threshold percent correct']=computeThresholdPercentCorrect
metric_types['tpc']=computeThresholdPercentCorrect
metric_names[computeThresholdPercentCorrect]='Threshold Percent Correct'

def computeEqualErrorRate (x, y, tags=None):
    # Find threshold that minimizes squared difference between FP and FN rates
    (best,bestT) = optimizeOverThresholds(computeErrorRateDiffSquared, False, x, y, tags)
    # For that threshold, return the percent correct
    x = threshold(x, bestT)
    return computePercentCorrect(x, y, tags)
metric_types['equalerrorrate']=computeEqualErrorRate
metric_types['equalerror']=computeEqualErrorRate
metric_types['equal errorrate']=computeEqualErrorRate
metric_types['equal error']=computeEqualErrorRate
metric_types['eer']=computeEqualErrorRate
metric_types['ee']=computeEqualErrorRate
metric_names[computeEqualErrorRate]='Equal Error Rate'

def optimizeOverBaselinesAndScales (metricFn, shouldMaximize, x, y, tags = None):
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
            value = metricFn(x, y, tags)
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

def optimizeOverThresholds (metricFn, shouldMaximize, x, y,numThresholds = None, tags = None):
    # Compute array of all useful thresholds
    # (x - eps for every x, and also max(x) + eps)
    x = np.array(x)
    eps = sys.float_info.epsilon

    if numThresholds is not None:
        thresholds = np.arange(min(x), max(x), (max(x)-min(x))/float(numThresholds))
    else:
        thresholds = x - eps
        thresholds = np.append(thresholds, max(x) + eps)

    if shouldMaximize:
        best = -float("inf")
        bestT = 0
    else:
        best = +float("inf")
        bestT = 0
    origX = x
    for t in thresholds:
        x = threshold(origX, t)
        value = metricFn(x, y, tags)
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

def computeOptimalF1 (x, y, numThresholds = 100, tags=None):
    return optimizeOverThresholds(computeF1, True, x, y, numThresholds, tags)
metric_types['optimalf1']=computeOptimalF1
metric_types['of1']=computeOptimalF1
metric_names[computeOptimalF1]='Optimal F1'

def computeBinarizedF1 (x, y, tags = None, t=0):
    x = threshold(x,t)
    return computeF1(x, y, tags)
metric_types['binarizedf1']=computeBinarizedF1
metric_types['bf1']=computeBinarizedF1
metric_names[computeBinarizedF1]='Binarized F1'

def computeF1 (x, y, tags = None):
    return computeF(x, y, tags, 0.5, 1)
metric_types['f1']=computeF1
metric_names[computeF1]='F1'

# Convert a log-likelihood ratio into a probability (around baseline b)
def convertLLRtoProb (x, b):
    return 1. / (1. + math.exp(- (x - b)))

def computeOptimalBalancedLogisticCost (x, y, tags = None):
    return optimizeOverBaselinesAndScales(computeBalancedLogisticCost, False, x, y, tags)
metric_types['optimalbalancedlogisticcost']=computeOptimalBalancedLogisticCost
metric_types['oblc']=computeOptimalBalancedLogisticCost
metric_names[computeOptimalBalancedLogisticCost]='Optimal Balanced Logistic Cost'

def computeOptimalBalancedExponentialCost (x, y, tags = None):
    return optimizeOverBaselinesAndScales(computeBalancedExponentialCost, False, x, y, tags)
metric_types['optimalbalancedexponentialcost']=computeOptimalBalancedExponentialCost
metric_types['obec']=computeOptimalBalancedExponentialCost
metric_names[computeOptimalBalancedExponentialCost]='Optimal Balanced Exponential Cost'

def computeBinarizedBalancedLogisticCost (x, y, tags = None):
    return computeBalancedLogisticCost(x, y, tags)
metric_types['binarizedbalancedlogisticcost']=computeBinarizedBalancedLogisticCost
metric_types['bblc']=computeBinarizedBalancedLogisticCost
metric_names[computeBinarizedBalancedLogisticCost]='Binarized Balanced Logistic Cost'

def computeBinarizedBalancedExponentialCost (x, y, tags = None):
    return computeBalancedExponentialCost(x, y, tags)
metric_types['binarizedbalancedexponentialcost']=computeBinarizedBalancedExponentialCost
metric_types['bbec']=computeBinarizedBalancedExponentialCost
metric_names[computeBinarizedBalancedExponentialCost]='Binarized Balanced Exponential Cost'

def computeBalancedLogisticCost (x, y, tags = None):
    return computeBalancedLogisticAndExponentialCosts(x, y, tags)[0]
metric_types['balancedlogisticcost']=computeBalancedLogisticCost
metric_types['blc']=computeBalancedLogisticCost
metric_names[computeBalancedLogisticCost]='Balanced Logistic Cost'

def computeBalancedExponentialCost (x, y, tags = None):
    return computeBalancedLogisticAndExponentialCosts(x, y, tags)[1]
metric_types['balancedexponentialcost']=computeBalancedExponentialCost
metric_types['bec']=computeBalancedExponentialCost
metric_names[computeBalancedExponentialCost]='Balanced Exponential Cost'

def computeBalancedLogisticAndExponentialCosts (x, y, tags = None):
    x = np.array(x, float)
    y = np.array(y, float)
    # If specified, limit ourselves to data for which tag == 1
    if tags is not None:
        tags = np.array(tags, float)
        x = x[pylab.find(tags == 1)]
        y = y[pylab.find(tags == 1)]
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
def compute2AFC (x, y, tags = None):
    x = np.array(x, float)
    y = np.array(y, float)
    if tags is not None:
        tags = np.array(tags, float)

        N = y.shape[0]
        c = np.unique(y)

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
        if (tags is not None) and tags[idxs0[k]] == 0:
            valid *= np.array(tags[idxs1] == 1)
        numPairs[k] = np.sum(valid)
                # if I get item x0[k] to represent category 0 in combination with
                # any randomly selected x1 item representing category 1 what's the probability that I
                # classify x0[k] correctly
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
