from lasagne.init import Initializer

__all__=[
	"MaskedInit"
]

class MaskedInit(Initializer):
    def __init__(self,initializer,mask):
        self.initializer = initializer
        self.mask=mask

    def sample(self,shape):
        s = self.initializer.sample(shape)
        return s*self.mask
