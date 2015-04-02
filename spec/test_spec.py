from xnn.spec import *
from xnn.spec.layers import *
from xnn.spec.train.trainer import *

import pprint


BATCHSIZE = 128
IMGWIDTH = 48
IMGHEIGHT = 48
IMGDIMS = IMGWIDTH*IMGHEIGHT


# -------------------------------------------------------------------- #
# First setup the data
# -------------------------------------------------------------------- #
input_names = ['patches']

aus = ChannelSet(name="someaus", type="action_units", type_weight=.5)
aus.add(Channel("AU1", negative_weight=.7, channel_weight=.4))
aus.add(Channel("AU2", negative_weight=.5, channel_weight=.8))
aus.add(Channel("AU4", negative_weight=.7, channel_weight=.4))

emos = ChannelSet(name="allemotions", type="emotions", type_weight=1.0)
emos.add(Channel("anger", negative_weight=.1, channel_weight=.7))
emos.add(Channel("contempt", negative_weight=.1, channel_weight=.9))

data_mgr = DataManager(input_names=input_names,
                       channel_sets=[aus, emos],
                       batch_size=128,
                       shuffle_batches=True)


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

# Training spec
train_settings = TrainerSettings(learning_rate=ConstantVal(0.01),
                                 momentum=ConstantVal(0.5),
                                 update=NesterovMomentum,
                                 weightdecay=L2(),
                                 epochs=100)
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





#import os
#from draw_net.draw_net import *
#
#nettypes = ['mlp', 'autoencoder', 'joint', 'semi']
#for nettype in nettypes:
#    mynet = eval(nettype)
#    layers=mynet.instantiate()
#    # draw_to_notebook(layers, verbose=True, output_shape=False)
#    draw_to_file(layers, "/tmp/%s.png" % nettype, verbose=True, output_shape=False)
#    os.system('open /tmp/%s.png' % nettype)

