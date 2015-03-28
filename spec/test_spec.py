from spec import *
import pprint
import json

# -------------------------------------------------------------------- #
# Basic nonlinear autoencoder (without weight-sharing)
# -------------------------------------------------------------------- #
autoencoder = modelSpec()
pixels = autoencoder.add(
    inputLayerSpec(shape=(48,48)))
hidden = autoencoder.add(
    denseLayerSpec(parent=pixels.name,
                   num_units=100,
                   nonlinearity=tanhSpec()))
recon = autoencoder.add(
    denseLayerSpec(parent=hidden.name,
                   num_units=48*48,
                   nonlinearity=linearSpec()))

autoencoder.bind_output(
    layername=recon.name,
    settings=outputSpec(
        loss=mseSpec(),
        target=ReconstructionTargetSpec(layer=pixels.name)
    )
)
print '---------------------------------'
print 'Autoencoder dict representation'
pprint.pprint(autoencoder.to_dict())
print 'Autoencoder instantiated representation'
pprint.pprint(autoencoder.instantiate())

# -------------------------------------------------------------------- #
# Basic mlp with softmax layers
# -------------------------------------------------------------------- #
mlp = modelSpec()
pixels = mlp.add(
    inputLayerSpec(shape=(48,48)))
hidden1 = mlp.add(
    denseLayerSpec(parent=pixels.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
hidden2 = mlp.add(
    denseLayerSpec(parent=hidden1.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
labels = mlp.add(
    denseLayerSpec(parent=hidden2.name,
                   num_units=20,
                   nonlinearity=softmaxSpec()))

mlp.bind_output(
    layername=labels.name,
    settings=outputSpec(
        loss=categorical_crossentropySpec(),
        target=ChannelsTargetSpec()
    )
)

print '---------------------------------'
print 'MLP dict representation'
pprint.pprint(mlp.to_dict())
print 'MLP instantiated representation'
pprint.pprint(mlp.instantiate())

# -------------------------------------------------------------------- #
# Joint nonlinear autoencoder+mlp (without weight-sharing)
# -------------------------------------------------------------------- #
joint = modelSpec()
pixels = joint.add(
    inputLayerSpec(shape=(48,48)))
hid1up = joint.add(
    denseLayerSpec(parent=pixels.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
encoder = joint.add(
    denseLayerSpec(parent=hid1up.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
hid1down = joint.add(
    denseLayerSpec(parent=encoder.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
# This is the pixel reconstruction layer
recon = joint.add(
    denseLayerSpec(parent=hid1down.name,
                   num_units=48*48,
                   nonlinearity=linearSpec()))
# This is the label layer for supervised learning
labels = joint.add(
    denseLayerSpec(parent=encoder.name,
                   num_units=10,
                   nonlinearity=softmaxSpec()))

# Make the recon layer reconstruct the pixels with .5 scaling of the gradients
joint.bind_output(
    layername=recon.name,
    settings=outputSpec(
        loss=mseSpec(),
        scale=0.5,
        target=ReconstructionTargetSpec(layer=pixels.name)
    )
)

# Make the label layer recognize the labels with full gradients
joint.bind_output(
    layername=labels.name,
    settings=outputSpec(
        loss=categorical_crossentropySpec(),
        target=ChannelsTargetSpec()
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
semi = modelSpec()
pixels = semi.add(
    inputLayerSpec(shape=(48,48)))
hid1up = semi.add(
    denseLayerSpec(parent=pixels.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
# Use sigmoid labels (like we do in practice)
labelencoder = semi.add(
    denseLayerSpec(parent=hid1up.name,
                   num_units=10,
                   nonlinearity=sigmoidSpec()))
# Make hidden encoder also use sigmoid for consistency
hidencoder = semi.add(
    denseLayerSpec(parent=hid1up.name,
                   num_units=100,
                   nonlinearity=sigmoidSpec()))
# This is the joint encoder layer with label predictions and hidden units
jointencoder = semi.add(
    concatLayerSpec(parents=[labelencoder.name, hidencoder.name]))
hid1down = semi.add(
    denseLayerSpec(parent=jointencoder.name,
                   num_units=100,
                   nonlinearity=rectifySpec()))
# This is the pixel reconstruction layer
recon = semi.add(
    denseLayerSpec(parent=hid1down.name,
                   num_units=48*48,
                   nonlinearity=linearSpec()))

# Make the recon layer reconstruct the pixels
semi.bind_output(
    layername=recon.name,
    settings=outputSpec(
        loss=mseSpec(),
        target=ReconstructionTargetSpec(layer=pixels.name)
    )
)

# Make the label encoder layer recognize the labels
semi.bind_output(
    layername=labelencoder.name,
    settings=outputSpec(
        loss=crossentropySpec(),
        target=ChannelsTargetSpec()
    )
)

print '---------------------------------'
print 'Semi-supervised autoencoder dict representation'
pprint.pprint(semi.to_dict())
print 'Semi-supervised autoencoder instantiated representation'
pprint.pprint(semi.instantiate())




#
# m2 = modelSpec()
# l0 = m2.add(inputLayerSpec(shape=(48,48)))
# l1 = m2.add(denseLayerSpec(parent=l0.name, **b_props))
# l2 = m2.add(concatLayerSpec(parents=[l0.name, l1.name]))
# l3 = m2.add(denseLayerSpec(parent=l2.name, num_units=100))
#
# m2.bind_output(layername=l3.name,
#                settings=outputSpec(
#                    loss=crossentropySpec(),
#                    eval_output_activation=linearSpec(), scale=1.0,
#                    target=ReconstructionTargetSpec(layer=m2.last().name)))
#

# -------------------------------------------------------------------- #

# out_props = outputSpec(loss=crossentropySpec(), eval_output_activation=linearSpec(), scale=1.0, target=ChannelsTargetSpec(separator=None))
#
#
# # using first and last functions
# m3 = modelSpec()
# m3.add(inputLayerSpec(shape=(48,48)))
# m3.add(denseLayerSpec(parent=m3.first(), **b_props))
# m3.add(concatLayerSpec(parents=[m3.first(), m3.last()]))
# m3.add(denseLayerSpec(parent=m3.last(), num_units=100), output_settings=out_props)
#
# # using explicit reference by layer name (where names are auto-generated)
# m4 = modelSpec()
# m4.add(inputLayerSpec(shape=(48,48)))
# m4.add(denseLayerSpec(parent=m4.get_layer("inputLayer_0"), **b_props))
# m4.add(concatLayerSpec(parents=[m4.get_layer("inputLayer_0"), m4.get_layer("denseLayer_1")]))
# m4.add(denseLayerSpec(parent=m4.get_layer("concatLayer_2"), num_units=100), output_settings=out_props)
#
# # using explicit reference by layer name (where names are manually specified)
# m5 = modelSpec()
# m5.add(inputLayerSpec(shape=(48,48)), name="inputLayer_0")
# m5.add(denseLayerSpec(parent=m5.get_layer("inputLayer_0"), **b_props), name="denseLayer_1")
# m5.add(concatLayerSpec(parents=[m5.get_layer("inputLayer_0"), m5.get_layer("denseLayer_1")]), name="concatLayer_2")
# m5.add(denseLayerSpec(parent=m5.get_layer("concatLayer_2"), num_units=100), name="denseLayer_3", output_settings=out_props)
#
#
# assert m1.to_dict()==m2.to_dict()==m3.to_dict()==m4.to_dict()==m5.to_dict()