from xnn.spec import *
from xnn.spec.layers import *
from xnn.spec.train.trainer import *

import pprint


BATCHSIZE = 128
MAXEPOCHS = 100
IMGWIDTH = 48
IMGHEIGHT = 48
IMGDIMS = IMGWIDTH*IMGHEIGHT
AGEBINS = [18, 25, 35, 45, 55, 65, 100] # right bin edges
DRAWNETS = True # only if pydot is installed and draw_net is present

# -------------------------------------------------------------------- #
# First setup the data
# -------------------------------------------------------------------- #
# TODO: Add something to specify the dataset/loader class/function (e.g. auload, "mnist"...)

input_names = ['patches']

aus = ChannelSet(name="someaus", type="action_units", set_weight=.5)
aus.add(Binary("AU1", channel_weight=.4, negative_weight=.7))
aus.add(Binary("AU2", channel_weight=.8))
aus.add(Binary("AU4", negative_weight=.7))

emos = ChannelSet(name="allemotions", type="emotions")
emos.add(Binary("anger"))
emos.add(Binary("contempt"))

age = ChannelSet(name="age", type="age")
age.add(Real("age", bins_to_weight=AGEBINS))

ethn = ChannelSet(name="ethnicities", type="ethnicities")
ethn.add(Real("asian"))
ethn.add(Real("black"))
ethn.add(Real("hispanic"))
ethn.add(Real("indian"))
ethn.add(Real("white"))

data_mgr = DataManager(input_names=input_names,
                       channel_sets=[aus, emos, age, ethn],
                       batch_size=128,
                       shuffle_batches=True)


# -------------------------------------------------------------------- #
# Sequential basic mlp with softmax layers
# -------------------------------------------------------------------- #
seqmlp = Sequential()
seqmlp.add(InputLayer, shape=(BATCHSIZE, IMGDIMS))
seqmlp.add(DenseLayer, num_units=100, nonlinearity=rectify())
seqmlp.add(DenseLayer, num_units=200, nonlinearity=rectify())
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
train_settings = TrainerSettings(
    update=NesterovMomentum(
        learning_rate=ConstantVal(0.01), momentum=ConstantVal(0.5)),
    weightcost=L2(ConstantVal(1e-5)),
    max_epochs=MAXEPOCHS)
trainer = Trainer(seqmlp, data_manager=data_mgr, default_settings=train_settings)



print '---------------------------------'
print 'MLP dict representation'
pprint.pprint(seqmlp.to_dict())
print 'MLP instantiated representation'
pprint.pprint(seqmlp.instantiate())
print 'Trainer dict representation'
pprint.pprint(trainer.to_dict())



# -------------------------------------------------------------------- #
# Basic mlp with softmax layers
# -------------------------------------------------------------------- #
mlp = Model()
pixels = mlp.add(InputLayer(shape=(BATCHSIZE, IMGDIMS)), name="pixels")
hidden1 = mlp.add(DenseLayer(parent=pixels.name, num_units=100,
                             nonlinearity=rectify()), name="hidden1")
hidden2 = mlp.add(DenseLayer(parent=hidden1.name, num_units=200,
                             nonlinearity=rectify()), name="hidden2")
labels = mlp.add(DenseLayer(parent=hidden2.name, num_units=aus.size(),
                            nonlinearity=softmax()), name="labels")

mlp.add_channel_set(aus)

mlp.bind_output(
    layername=labels.name,
    settings=Output(
        loss=categorical_crossentropy(),
        target=ChannelsTarget(channelsets=[aus])
    )
)

mlp.bind_param_update_settings(layername=hidden1.name,
                               settings=ParamUpdateSettings(
                                   learning_rate=ConstantVal(0.001),
                                   momentum=ConstantVal(0.5)))

trainer = Trainer(mlp, data_manager=data_mgr, default_settings=train_settings)

print '---------------------------------'
print 'MLP dict representation'
pprint.pprint(mlp.to_dict())
print 'MLP instantiated representation'
pprint.pprint(mlp.instantiate())


# -------------------------------------------------------------------- #
# Basic nonlinear autoencoder (without weight-sharing)
# -------------------------------------------------------------------- #
autoencoder = Model()
pixels = autoencoder.add(InputLayer(shape=(BATCHSIZE,IMGDIMS)), name="pixels")
hidden = autoencoder.add(DenseLayer(parent=pixels.name, num_units=100,
                                    nonlinearity=tanh()), name="hidden")
recon = autoencoder.add(DenseLayer(parent=hidden.name, num_units=IMGDIMS,
                                   nonlinearity=linear()), name="recon")

autoencoder.bind_output(
    layername=recon.name,
    settings=Output(
        loss=mse(),
        target=ReconstructionTarget(layer=pixels.name)
    )
)
print '---------------------------------'
print 'Autoencoder dict representation'
pprint.pprint(autoencoder.to_dict())
print 'Autoencoder instantiated representation'
pprint.pprint(autoencoder.instantiate())


# -------------------------------------------------------------------- #
# Joint nonlinear autoencoder+mlp (without weight-sharing)
# -------------------------------------------------------------------- #
joint = Model()
pixels = joint.add(InputLayer(shape=(BATCHSIZE,IMGDIMS)), name="pixels")
hid1up = joint.add(DenseLayer(parent=pixels.name, num_units=100,
                              nonlinearity=rectify()), name="hid1up")
encoder = joint.add(DenseLayer(parent=hid1up.name, num_units=100,
                               nonlinearity=rectify()), name="encoder")
hid1down = joint.add(DenseLayer(parent=encoder.name, num_units=100,
                                nonlinearity=rectify()), name="hid1down")
# This is the pixel reconstruction layer
recon = joint.add(DenseLayer(parent=hid1down.name, num_units=IMGDIMS,
                             nonlinearity=linear()), name="recon")
# This is the label layer for supervised learning
labels = joint.add(DenseLayer(parent=encoder.name, num_units=emos.size(),
                              nonlinearity=softmax()), name="labels")

joint.add_channel_set(emos)

# Make the recon layer reconstruct the pixels with .5 scaling of the gradients
joint.bind_output(
    layername=recon.name,
    settings=Output(
        loss=mse(),
        scale=0.5,
        target=ReconstructionTarget(layer=pixels.name)
    )
)

# Make the label layer recognize the labels with full gradients
joint.bind_output(
    layername=labels.name,
    settings=Output(
        loss=categorical_crossentropy(),
        target=ChannelsTarget(channelsets=[emos])
    )
)

print '---------------------------------'
print 'Joint autoencoder+MLP dict representation'
pprint.pprint(joint.to_dict())
print 'Joint autoencoder+MLP instantiated representation'
pprint.pprint(joint.instantiate())


# -------------------------------------------------------------------- #
# Semi-supervised autoencoder (without weight-sharing)
# -------------------------------------------------------------------- #
semi = Model()
pixels = semi.add(
    InputLayer(shape=(BATCHSIZE,IMGDIMS)), name="pixels")
hid1up = semi.add(
    DenseLayer(parent=pixels.name,
               num_units=100,
               nonlinearity=rectify()), name="hid1up")
# Use sigmoid labels (like we do in practice)
labelencoder = semi.add(
    DenseLayer(parent=hid1up.name,
               num_units=emos.size(),
               nonlinearity=sigmoid()), name="labelencoder")
# Make hidden encoder also use sigmoid for consistency
hidencoder = semi.add(
    DenseLayer(parent=hid1up.name,
               num_units=100,
               nonlinearity=sigmoid()), name="hidencoder")
# This is the joint encoder layer with label predictions and hidden units
jointencoder = semi.add(
    ConcatLayer(parents=[labelencoder.name, hidencoder.name]), name="jointencoder")
hid1down = semi.add(
    DenseLayer(parent=jointencoder.name,
               num_units=100,
               nonlinearity=rectify()), name="hid1down")
# This is the pixel reconstruction layer
recon = semi.add(
    DenseLayer(parent=hid1down.name,
               num_units=IMGDIMS,
               nonlinearity=linear()), name="recon")

semi.add_channel_set(emos)

# Make the recon layer reconstruct the pixels
semi.bind_output(
    layername=recon.name,
    settings=Output(
        loss=mse(),
        target=ReconstructionTarget(layer=pixels.name)
    )
)

# Make the label encoder layer recognize the labels
semi.bind_output(
    layername=labelencoder.name,
    settings=Output(
        loss=crossentropy(),
        target=ChannelsTarget(channelsets=[emos])
    )
)

print '---------------------------------'
print 'Semi-supervised autoencoder dict representation'
pprint.pprint(semi.to_dict())
print 'Semi-supervised autoencoder instantiated representation'
pprint.pprint(semi.instantiate())




if DRAWNETS:
    import os
    from draw_net.draw_net import *

    nettypes = ['mlp', 'autoencoder', 'joint', 'semi']
    for nettype in nettypes:
        mynet = eval(nettype)
        layers=mynet.instantiate()
        # draw_to_notebook(layers, verbose=True, output_shape=False)
        draw_to_file(layers, "/tmp/%s.png" % nettype, verbose=True, output_shape=False)
        os.system('open /tmp/%s.png' % nettype)



