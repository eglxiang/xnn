#!/usr/bin/env python
import xnn
from mnist import *

class Cond(xnn.experiments.ExperimentCondition):
    def __init__(self):
        self.learning_rate = 0.1
        self.hiddenunits   = 500
        self.droptype      = 'standard' 
        self.hiddropout    = 0.5


def set_up_experiment():
    expt = xnn.experiments.Experiment(name='mnist mlp',default_condition=Cond())
    expt.add_group('std drop')
    expt.add_group('gauss drop')
    expt.add_factor('learning_rate',[0.001,0.1])
    expt.add_factor('hiddenunits',[20,200,500])
    expt.add_factor('droptype','gauss',groupname='gauss drop')
    expt.add_factor('droptype','standard',groupname='std drop')
    expt.add_factor('hiddropout',[.3,1.5],groupname='gauss drop')
    expt.add_factor('hiddropout',[.3,.75],groupname='std drop')

    return expt

def run_experiment():
    #--------
    # Set up data, experiment, and metrics 
    #--------
    dataset = load_dataset()
    expt = set_up_experiment()

    metrics = [
        ('l_out', Metric(computeCategoricalCrossentropy, "y", aggregation_type="mean"), 'min'),
        ('l_out', Metric(computeOneHotAccuracy, "y", aggregation_type="mean"), 'max')
    ]

    trainbatchit = iterate_minibatches(dataset, BATCHSIZE, 'train')
    validbatchit = iterate_minibatches(dataset, BATCHSIZE, 'valid')

    #--------
    # Run all conditions in experiment, and store results
    #--------

    for conddict in expt.get_all_condition_iterator():
        print "\nRunning condition %d\n"%conddict['condition_num']
        c = conddict['condition']
        m = build_mlp(numhidunits=c.hiddenunits,hiddropout=c.hiddropout,dropout_type=c.droptype)
        t = set_up_trainer(m,learning_rate=c.learning_rate)
        loop = xnn.training.Loop(t,trainbatchit,validbatchit,metrics,plotmetricmean=False)
        metvals = loop(1)
        expt.add_results(conddict['condition_num'],metvals)


    #--------
    # Interpret results
    #--------

    #get results for standard vs gaussian dropout
    std_nums = expt.get_condition_numbers(fixed_dict={'droptype':'standard'})
    gss_nums = expt.get_condition_numbers(fixed_dict={'droptype':'gauss'})

    print "Standard dropout"
    bsp = None
    for sn in std_nums:
        cc,pc = expt.results[sn]
        bsp = pc if (bsp is None or pc > bsp) else bsp
        print "%d: %0.3f"%(sn,pc)  
    print "Gaussian dropout"
    bgp = None
    for sn in gss_nums:
        cc,pc = expt.results[sn]
        bgp = pc if (bgp is None or pc > bgp) else bgp
        print "%d: %0.3f"%(sn,pc)  

    print "Best Standard PC: %0.3f"%bsp
    print "Best Gaussian PC: %0.3f"%bgp

    #get results for hidden unit size 
    bhu = [None]*3
    for i,hu in enumerate([20,200,500]):
        print "%d hidden units"%hu
        nums = expt.get_condition_numbers(fixed_dict={'hiddenunits':hu})
        for sn in nums:
            cc,pc = expt.results[sn]
            bhu[i] = pc if (bhu[i] is None or pc > bhu[i]) else bhu[i] 
            print "%d: %0.3f"%(sn,pc)  
    for bh,hu  in zip(bhu,[20,200,500]):
        print "Best %d HU PC: %0.3f"%(hu,bh)

if __name__ == '__main__':
    run_experiment()
