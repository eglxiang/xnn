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

It is encouraged install the libraries in a virtual environment. Information on virtual environments can be found at https://virtualenv.pypa.io/en/latest/

To install the libraries required for XNN, run the following commands::

	cd xnn
	pip install -r requirements.txt

To add XNN to your python path, run the following command::

	export PYTHONPATH=$PYTHONPATH:./

XNN is closely tied with the Lasagne library. For more information on Lasagne, go to http://lasagne.readthedocs.org/

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

