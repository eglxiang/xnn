import xnn
from xnn import utils
import numpy as np
import theano
import theano.tensor as T

def test_digitize():
	bins = [-1.5,-0.5,0.5,1.5]
	a = [-5,-1.5,5,1.5,np.nan,0.75,0,-0.75]
	expected_out = [0, 1, 4, 4, 4, 3, 2, 1]
	x=T.vector()
	actual_out = utils.theano_digitize(x,bins).eval({x:a})
	assert np.all(actual_out == expected_out)

def test_one_hot():
	a = [0,1,3,9,-1,6,2,2,10]
	expected_out = [[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]]
	actual_out = utils.numpy_one_hot(a)
	assert np.all(actual_out == expected_out)

def test_nan_funcs():
	a = [[1,2],[-1,np.nan],[0,0]]
	expected_mean = [0,1]
	expected_max = [2,-1,0]
	expected_sum = 2
	x = T.matrix()
	actual_mean = utils.Tnanmean(x,axis=0).eval({x:a})
	actual_max = utils.Tnanmax(x,axis=1).eval({x:a})
	actual_sum = utils.Tnansum(x).eval({x:a})
	assert np.all(actual_mean == expected_mean)
	assert np.all(actual_max == expected_max)
	assert np.all(actual_sum == expected_sum)

if __name__ == '__main__':
	test_digitize()
	test_one_hot()
	test_nan_funcs()
