import xnn
from xnn.metrics.metricsSuite import *
import numpy as np

binpred = np.array([[1,0,0],[0,1,0],[0,0,0],[.4,1,0],[1.,1.,1.]])
bintarg = np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[np.nan,np.nan,np.nan]])
probpred= np.array([[.2,.8,0],[.8,.2,0],[1,0,0],[0,1,0],[0,0,1]])
betterprobpred=np.array([[.6,.4,0],[.8,.2,0],[1,0,0],[.5,.5,0],[0,0,1]])
probtarg= np.array([[.8,.2,0],[.8,.2,0],[0,1,0],[0,0,1],[np.nan,np.nan,np.nan]])
regpred = np.array([[10],[9],[2],[1]])
regtarg = np.array([[5],[5],[5],[5]])

td = dict(bintarg=bintarg,probtarg=probtarg,regtarg=regtarg)


def test_binarizedBER():
    bbe = xnn.metrics.Metric('bbeb','bintarg',aggregation_type='sum')
    # print "bbeb", bbe(binpred,td)
    assert np.allclose(bbe(binpred,td),0.75)
    bbec = xnn.metrics.Metric('bbec','probtarg',aggregation_type='none')
    # print "bbec", bbec(probpred,td)
    assert np.allclose(bbec(probpred,td),[0.5,0.83333333,0.5])
    # print "betterbbec", bbec(betterprobpred,td)
    assert np.allclose(bbec(betterprobpred,td),[0.25,.5,.5])

def test_cce_and_KL():
    cce = xnn.metrics.Metric('cce','probtarg',aggregation_type='none')
    # print "cce",cce(probpred,td)
    assert np.allclose(cce(probpred,td)[0:4], [1.33217907, 0.50040239, 50., 50.])
    assert np.isnan(cce(probpred,td)[-1])
    # print "bettercce",cce(betterprobpred,td)
    assert np.allclose(cce(betterprobpred,td)[0:4], [0.59191859, 0.50040239, 50., 50.])
    assert np.isnan(cce(betterprobpred,td)[-1])
    cceMean = xnn.metrics.Metric('cce','probtarg',aggregation_type='mean')
    # print "cceMean",cceMean(probpred,td)
    assert np.allclose(cceMean(probpred,td),25.4581)
    kl = xnn.metrics.Metric('kl','probtarg',aggregation_type='none')
    # print "kl",kl(probpred,td)
    assert np.allclose(kl(probpred,td),[  0.83177662, 0., 50., 50., 0.])
    klMean = xnn.metrics.Metric('kl','probtarg',aggregation_type='mean')
    # print "klMean",klMean(probpred,td)
    assert np.allclose(klMean(probpred,td),20.1664)

def test_optimized_threshold():
    obeMean = xnn.metrics.Metric('obe','bintarg',aggregation_type='none')
    # print 'optimalBE',obeMean(binpred,td)
    assert np.allclose(obeMean(binpred,td)[0],0.25)
    assert np.allclose(obeMean(binpred,td)[1],0.0)
    
    ofoMean = xnn.metrics.Metric('of1','bintarg',aggregation_type='none')
    #print 'optimalF1',ofoMean(binpred,td)
    assert np.allclose(ofoMean(binpred,td)[0],0.75)
    assert np.allclose(ofoMean(binpred,td)[1],0.0)

def test_regression_metrics():
    mse = xnn.metrics.Metric('se','regtarg',aggregation_type='mean')
    mae = xnn.metrics.Metric('ae','regtarg',aggregation_type='mean')
    sse = xnn.metrics.Metric('se','regtarg',aggregation_type='sum')
    assert np.allclose(mse(regpred,td),21.5)
    assert np.allclose(mae(regpred,td),4.0)
    assert np.allclose(sse(regpred,td),86)
    #print 'mse',mse(regpred,td)
    #print 'mae',mae(regpred,td)
    #print 'sse',sse(regpred,td)

def test_confmat():
    cm = xnn.metrics.Metric(computeConfusionMatrix,'probtarg',aggregation_type='none')
    #print 'cm',cm(probpred,td)
    np.allclose(cm(probpred,td),[[2,2,0],[0,0,0],[0,0,0]])
    cmD = xnn.metrics.Metric(computeConfusionMatrix,'probtarg',aggregation_type=confMatAggregate)
    np.allclose(cmD(probpred,td)[0],[[2,2,0],[0,0,0],[0,0,0]])
    np.allclose(cmD(probpred,td)[1],0.16666666666666666)
    #print cmD(probpred,td)
    #print cmD.to_dict()


if __name__=="__main__":
    test_binarizedBER()
    test_cce_and_KL()
    test_optimized_threshold()
    test_regression_metrics()
    test_confmat()
