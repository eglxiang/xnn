# from spec.specbridge import *
from spec import *

# All 5 of the following models should be equal

b_props = dict(
    num_units=100, Winit=UniformSpec(range=(1,3)), binit=ConstantSpec(val=0.0), nonlinearity=linearSpec()#LeakyRectifySpec(leakiness=0.01)
)

# heterogenous
m1 = modelSpec()
l0 = m1.add(inputLayerSpec(shape=(48,48)))
m1.add(denseLayerSpec(parent=l0, **b_props))
m1.add(concatLayerSpec(parents=[m1.first(), m1.last()]))
m1.add(denseLayerSpec(parent=m1.get_layer("concatLayer_2"), num_units=100))

# using passed back layer objects
m2 = modelSpec()
l0 = m2.add(inputLayerSpec(shape=(48,48)))
l1 = m2.add(denseLayerSpec(parent=l0, **b_props))
l2 = m2.add(concatLayerSpec(parents=[l0, l1]))
l3 = m2.add(denseLayerSpec(parent=l2, num_units=100))

# using first and last functions
m3 = modelSpec()
m3.add(inputLayerSpec(shape=(48,48)))
m3.add(denseLayerSpec(parent=m3.first(), **b_props))
m3.add(concatLayerSpec(parents=[m3.first(), m3.last()]))
m3.add(denseLayerSpec(parent=m3.last(), num_units=100))

# using explicit reference by layer name (where names are auto-generated)
m4 = modelSpec()
m4.add(inputLayerSpec(shape=(48,48)))
m4.add(denseLayerSpec(parent=m4.get_layer("inputLayer_0"), **b_props))
m4.add(concatLayerSpec(parents=[m4.get_layer("inputLayer_0"), m4.get_layer("denseLayer_1")]))
m4.add(denseLayerSpec(parent=m4.get_layer("concatLayer_2"), num_units=100))

# using explicit reference by layer name (where names are manually specified)
m5 = modelSpec()
m5.add(inputLayerSpec(shape=(48,48)), name="inputLayer_0")
m5.add(denseLayerSpec(parent=m5.get_layer("inputLayer_0"), **b_props), name="denseLayer_1")
m5.add(concatLayerSpec(parents=[m5.get_layer("inputLayer_0"), m5.get_layer("denseLayer_1")]), name="concatLayer_2")
m5.add(denseLayerSpec(parent=m5.get_layer("concatLayer_2"), num_units=100), name="denseLayer_3")


assert m1.to_dict()==m2.to_dict()==m3.to_dict()==m4.to_dict()==m5.to_dict()