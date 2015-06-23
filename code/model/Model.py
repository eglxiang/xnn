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

    def addDenseDropStack(self,parent_layer,num_hidden_list=None,drop_p_list=None,nonlin_list=None,namebase=None):
        pl = parent_layer
        if namebase is None:
            namebase="l_"
        for i in xrange(len(num_hidden_list)):
            nhu = num_hidden_list[i]
            if drop_p_list is not None:
                p = drop_p_list[i]
            else:
                # default dropout value
                p = 0.5
            if nonlin_list is not None:
                nl = nonlin_list[i]
            else:
                # default nonlinearity
                nl = lasagne.nonlinearities.rectify
            denselayer = lasagne.layers.dense.DenseLayer(pl,num_units=nhu,nonlinearity=nl)
            self.addLayer(denselayer,name=namebase+'_dense_'+str(i))
            droplayer  = lasagne.layers.noise.DropoutLayer(denselayer,p=p)
            self.addLayer(droplayer,name=namebase+'_drop_'+str(i))
            pl = droplayer
        return pl

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
            if type(l) == lasagne.layers.dense.DenseLayer:
                ldict['nonlinearity'] = l.nonlinearity
            ls.append(ldict)

        d['layers']=ls
        d['name']=self.name
        return d





