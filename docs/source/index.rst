.. xnn documentation master file, created by
   sphinx-quickstart on Mon Jul 20 14:57:50 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xnn's documentation!
================================

############
Installation
############
Clone repository with following command::

	git clone ssh://git@stash.emotient.local/res/xnn.git

It is encouraged to install the libraries in a virtual environment. 
To do so, enter this command to create a virtualenvironment and activate it:

```
virtualenv venv
. venv/bin/activate
```

More information on virtual environments can be found at https://virtualenv.pypa.io/en/latest/

To install the libraries required for XNN, run the following commands::

	cd xnn
	pip install --upgrade -r requirements.txt

To add XNN to your python path, run the following command where <DIR_ABOVE_XNN> is the path to the dir above the xnn directory::

	export PYTHONPATH=$PYTHONPATH:<DIR_ABOVE_XNN>/

XNN is closely tied with the Lasagne library. For more information on Lasagne, go to http://lasagne.readthedocs.org/

##########################################
Verifying installation and testing library
##########################################
Run nosetests to ensure that all tests pass within your installation environment.
Note that the first time you run nosetests with a fresh xnn installation it will take a few minutes to compile various theano functions. Go grab a coffee.

To run nosetests using the CPU, run the following command from the root directory of the xnn repository::

    THEANO_FLAGS=device=cpu,floatX=float32 nosetests

To run nosetests using the GPU, replace the device=cpu with device=gpu in the command above.

The nosetests output should indicate that all tests have passed successfully.
If any tests fail, you can obtain more information by running nosetests with the -v flag for verbose output.

Note that it is good practice to run nosetests when you make changes to the library to ensure that nothing breaks.

#############################
Running the included examples
#############################
We will assume you have already run through and verified the XNN installation.
Now, enter the ``examples`` folder and run the mnist.py example script::

    cd examples
    # here we run on GPU
    THEANO_FLAGS=device=gpu,floatX=float32 python mnist.py


If everything is set up correctly, you will get an output like the following::

    Using gpu device 0: GeForce GT 750M
    ==========
    Epoch 0 / 499 -- 3.78 seconds
    Expected Finish: Fri, 12:34:42 PM
    ----------
    Training total cost                                               :   1.2393 (best 1.2393 at epoch 0)
    ----------
    Categorical Crossentropy                      l_out               :   0.5544 (best 0.5544 at epoch 0)
    Percent Correct                               l_out               :   0.8480 (best 0.8480 at epoch 0)
    ==========
    Epoch 1 / 499 -- 1.91 seconds
    Expected Finish: Fri, 12:23:50 PM
    ----------
    Training total cost                                               :   0.5670 (best 0.5670 at epoch 1)
    ----------
    Categorical Crossentropy                      l_out               :   0.4459 (best 0.4459 at epoch 1)
    Percent Correct                               l_out               :   0.8880 (best 0.8880 at epoch 1)
    ==========
    Epoch 2 / 499 -- 1.86 seconds
    Expected Finish: Fri, 12:20:17 PM
    ----------
    Training total cost                                               :   0.4661 (best 0.4661 at epoch 2)
    ----------
    Categorical Crossentropy                      l_out               :   0.3973 (best 0.3973 at epoch 2)
    Percent Correct                               l_out               :   0.9000 (best 0.9000 at epoch 2)
    ==========
    ...

Run the script with python mnist.py --help for more information.

#######
Modules
#######
.. toctree::
   :maxdepth: 2

   modules/xnn.data
   modules/xnn.experiments
   modules/xnn.init
   modules/xnn.layers
   modules/xnn.metrics
   modules/xnn.model
   modules/xnn.nonlinearities
   modules/xnn.objectives
   modules/xnn.training
   modules/xnn.utils

##################
Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

