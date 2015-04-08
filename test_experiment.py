from xnn.spec import *
from xnn.spec.layers import *
from xnn.spec.train.trainer import *
import os.path

EXPERIMENT_ID = "exp1"
# EXPERIMENT_ID = os.path.basename(__file__).rsplit('.',1)[0]
IMGWIDTH = 48
IMGHEIGHT = 48

class Condition(ExperimentCondition):
    def __init__(self):
        self.batchsize = 128
        self.shuffle = True
        self.maxepoch = 100
        self.imgdims = IMGWIDTH*IMGHEIGHT
        self.agebins = [18, 25, 35, 45, 55, 65, 100]
        self.auweight = .5
        self.emoweight = 1.0
        self.numhid1 = 100
        self.numhid2 = 50
        self.nonlin1 = rectify()
        self.nonlin2 = rectify()
        self.update = NesterovMomentum(learning_rate=ConstantVal(0.01), momentum=ConstantVal(0.5))
        self.wc = L2(ConstantVal(1e-5))

def create_spec(cond=Condition()):
    # -------------------------------------------------------------------- #
    # First setup the data
    # -------------------------------------------------------------------- #
    # TODO: Add something to specify the dataset/loader class/function (e.g. auload, "mnist"...)

    input_names = ['patches']

    aus = ChannelSet(name="someaus", type="action_units", set_weight=cond.auweight)
    aus.add(Binary("AU1", channel_weight=.4, negative_weight=.7))
    aus.add(Binary("AU2", channel_weight=.8))
    aus.add(Binary("AU4", negative_weight=.7))

    emos = ChannelSet(name="allemotions", type="emotions", set_weight=cond.emoweight)
    emos.add(Binary("anger"))
    emos.add(Binary("contempt"))

    age = ChannelSet(name="age", type="age")
    age.add(Real("age", bins_to_weight=cond.agebins))

    ethn = ChannelSet(name="ethnicities", type="ethnicities")
    ethn.add(Real("asian"))
    ethn.add(Real("black"))
    ethn.add(Real("hispanic"))
    ethn.add(Real("indian"))
    ethn.add(Real("white"))

    data_mgr = DataManager(input_names=input_names,
                           channel_sets=[aus, emos, age, ethn],
                           batch_size=cond.batchsize,
                           shuffle_batches=cond.shuffle)


    # -------------------------------------------------------------------- #
    # Sequential basic mlp with softmax layers
    # -------------------------------------------------------------------- #
    seqmlp = Sequential()
    seqmlp.add(InputLayer, shape=(cond.batchsize, cond.imgdims))
    seqmlp.add(DenseLayer, num_units=cond.numhid1, nonlinearity=cond.nonlin1)
    seqmlp.add(DenseLayer, num_units=cond.numhid2, nonlinearity=cond.nonlin2)
    seqmlp.add(DenseLayer, num_units=aus.size(), nonlinearity=softmax(), name="labels")

    seqmlp.add_channel_set(aus)

    seqmlp.bind_output(
        layername="labels",
        settings=Output(
            loss=categorical_crossentropy(),
            target=ChannelsTarget(channelsets=[aus])
        )
    )

    # Training spec
    train_settings = TrainerSettings(update=cond.update,
                                     weightdecay=cond.wc,
                                     max_epochs=cond.maxepoch)
    trainer = Trainer(seqmlp, data_manager=data_mgr, default_settings=train_settings)
    return trainer


expt = Experiment(EXPERIMENT_ID, Condition())
expt.add_factor("batchsize", [32, 64, 128])
expt.add_factor("numhid1", [1024,2048])
expt.add_factor("numhid2", [512,1024])
expt.add_factor("nonlin1", [rectify(), sigmoid()])
expt.add_factor("nonlin2", [rectify(), sigmoid()])

import pprint

print 'num conditions:', expt.get_num_conditions()
print 'conditions:'
pprint.pprint(expt.get_all_conditions_changes())

print 'condition(2)'
cond = expt.get_nth_condition(2)
trainer = create_spec(cond=cond)
pprint.pprint(trainer.to_dict())