import theano.tensor as T

objective_types={}
def mse(x, t, **kwargs):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.
    """
    return (x-t) ** 2

objective_types['mse'] = mse

def crossentropy(x, t, **kwargs):
    """Calculates the binary crossentropy mean across all dimentions,
    i.e.  feature dimension AND minibatch dimension.
    """
    return T.nnet.binary_crossentropy(x, t)

objective_types['crossentropy'] = crossentropy
objective_types['cross entropy'] = crossentropy
objective_types['ce'] = crossentropy

def categorical_crossentropy(x, t, **kwargs):
    return T.nnet.categorical_crossentropy(x, t)

objective_types['categorical crossentropy'] = categorical_crossentropy
objective_types['softmax crossentropy'] = categorical_crossentropy
objective_types['cce'] = categorical_crossentropy

def kl_divergence(x, t, eps=1e-08, **kwargs):
    """Calculates the KL-divergence between a true distribution t and the predictive distribution x
    Useful for comparing predicted probabilities to true probabilities (when labels are not necessarily binary).
    Can also use crossentropy or categorical_crossentropy for this (depending on sigmoid or softmax classes).
    KL-divergence has the property that when the predictions match the targets perfectly the divergence is 0.
    """
    lograt = T.log(t+eps) - T.log(x+eps)
    return T.sum(t * lograt, axis=1)

objective_types['kldivergence'] = kl_divergence
objective_types['kl divergence'] = kl_divergence
objective_types['kld'] = kl_divergence

def hinge_loss(x, t, **kwargs):
    t_ = T.switch(T.eq(t, 0), -1, 1)
    threshold = kwargs['threshold'] if 'threshold' in kwargs.keys() else 0
    scores = 1 - (t_ * x)
    return T.maximum(0, scores - threshold)

objective_types['hingeloss'] = hinge_loss
objective_types['hinge loss'] = hinge_loss
objective_types['hl'] = hinge_loss

def squared_hinge_loss(x, t, **kwargs):
    loss = hinge_loss(x, t, **kwargs)
    gamma = kwargs['gamma'] if 'gamma' in kwargs.keys() else 2.0
    return 1.0/(2.0 * gamma) * loss**2

objective_types['squaredhingeloss'] = squared_hinge_loss
objective_types['squared hinge loss'] = squared_hinge_loss
objective_types['shl'] = squared_hinge_loss


class Objective(object):
    def __init__(self, loss_function=objective_types['mse'], **kwargs):
        try:
            loss_function=loss_function.lower()
            self.loss_function = objective_types[loss_function]
        except:
            self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.settings = kwargs

    def handle_missing(self, loss, t, **kwargs):
        # Handle missing outputs by setting corresponding cost to 0
        if kwargs.has_key('has_missing') and kwargs['has_missing'] is not None:
            missing_indicator=kwargs['has_missing']
            loss *= T.neq(t, missing_indicator)  # This sets to zero all elements where Y == -1
        return loss
    
    def apply_reweightings(self, loss, t, **kwargs):
        # Reweight gradients for positive and/or negative examples
        # depending on specified weightings
        reweight = False
        posMask = t
        negMask = T.eq(posMask, 0)
        if 'negative_weighting' in kwargs.keys() and kwargs['negative_weighting'] is not None:
            negMask *= T.repeat(kwargs['negative_weighting'], t.shape[0], axis=0)
            reweight = True
        if reweight is True:
            loss *= (posMask + negMask)
    
        # Reweight gradients for different channels depending on specified weightings
        if 'channel_weighting' in kwargs.keys() and kwargs['channel_weighting'] is not None:
            loss *= T.repeat(kwargs['channel_weighting'], t.shape[0], axis=0)
        if 'type_weighting' in kwargs.keys() and kwargs['type_weighting'] is not None:
            loss *= T.repeat(kwargs['type_weighting'], t.shape[0], axis=0)
        return loss  

    def get_loss(self, input=None, target=None, **kwargs):
        network_output = input
        if target is None:
            target = self.target_var
            
        loss = self.loss_function(network_output, target, **dict(self.settings.items()+kwargs.items()))
        loss = self.handle_missing(loss, target, **kwargs)
        loss = self.apply_reweightings(loss, target, **kwargs)
        # return loss.sum()
        # return loss.mean(axis=0).sum()
        # return loss.mean(axis=0).sum()# T.sum(T.mean(loss,axis=0))
        return T.mean(loss)
