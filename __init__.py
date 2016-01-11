from lasagne import regularization,theano_extensions,updates
import model
import layers
import experiments
import data
import metrics
import training
import nonlinearities
import utils
import objectives
import init

######## HACK THAT MAY NOT BE NECESSARY IN THE FUTURE ########
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', '.*Glorot.*')
##############################################################
