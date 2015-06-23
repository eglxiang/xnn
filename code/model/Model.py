import lasagne

class Model():
    def __init__(self,name=None):
        self.name=name
        self.layers = {}

    def addLayer(self,layer,name=None):
        if name is None and layer.name is not None:
            name = layer.name
        if layer.name is None and name is not None:
            layer.name = name
        if (layer.name is None and name is None) or (name!=layer.name):
            raise Exception("Layer must have a consistent name")
        self.layers[name]=layer
        return layer

    def to_dict(self):
        d = {}
        ls = []
        for lname,l in self.layers.iteritems():
            if type(l) == lasagne.layers.input.InputLayer:
                ls.append(dict(name=lname,shape=l.shape,layer_type=type(l)))
                continue
            iln = l.input_layer.name if l.input_layer is not None else None
            ldict = dict(name=lname,
                           input_layer=iln,
                           input_shape=l.input_shape,
                           output_shape=l.output_shape,
                           layer_type=type(l))
            if type(l) == lasagne.layers.noise.DropoutLayer:
                ldict['p'] = l.p
            ls.append(ldict)

        d['layers']=ls
        d['name']=self.name
        return d





