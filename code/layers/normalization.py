from lasagne.layers.base import Layer
import theano.tensor as T


__all__ = [
    "ContrastNormLayer",
]


class ContrastNormLayer(Layer):
    """
    Implements per image contrast normalization
    """
    def __init__(self, input_layer, norm_type="mean_var", **kwargs):
        super(ContrastNormLayer, self).__init__(input_layer, **kwargs)
        if norm_type not in [None, "mean_var"]:
            raise Exception("norm_type %s Not implemented!" % norm_type)
        self.norm_type = norm_type

    def get_output_for(self, input, **kwargs):
        output_shape = input.shape
        if input.ndim > 2:
            input = T.flatten(input, 2)
        if self.norm_type == "mean_var":
            input -= T.mean(input, axis=1, keepdims=True)
            input /= T.std(input, axis=1, keepdims=True)
        input = input.reshape(output_shape)
        return input
