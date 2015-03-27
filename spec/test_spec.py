from spec import *

b_props = dict(
    num_units=100, Winit=UniformSpec(range=(1,3)), binit=ConstantSpec(val=0.0), nonlinearity=LeakyRectifySpec(leakiness=0.01)
)

m = modelSpec()
m.add(inputLayerSpec(name="in0", shape=(48,48)))
m.add(denseLayerSpec(name="d1", parent="in0", **b_props))
m.add(concatLayerSpec(name="c2", parents=["in0", "d1"]))
m.add(denseLayerSpec(name="o3", parent="c2", num_units=100))

# TODO:
# Currently the LayerSpec object expects a string to be passed to the constructor to refer to a parent.
# On the other hand, the underlying lasagne layer objects expect parent layers to be passed to their consructors.
# Thus layer objects in lasagne are not independent entities but keep track of their predecessors.
# In our spec library, the modelspec is a container class that explicitly keeps track of layers but may be redundant.
# For instance, instead of specifying the model as above, we could do so with only using layer object like this:
#   in0 = inputLayerSpec(name="in0", shape=(48,48))
#   d1 = denseLayerSpec(name="d1", parent=in0, **b_props)
#   c2 = concatLayerSpec(name="c2", parents=[in0, d1])
#   o3 = denseLayerSpec(name="o3", parent=c2, num_units=100)
# However, we may want to keep track of a machine more explicitly,
# especially if we have multiple branching output layers.
