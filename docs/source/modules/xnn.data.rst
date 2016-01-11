Data 
================

Data in xnn is passed between modules as a dictionary with
:class:`numpy.ndarray` as values.  As such, there is no special class for
containing data, and different data sources can be handled flexibly as long as
they return a dictionary.  In this module, we provide some convenience classes
for dealing with a particular form of data, but this is not intended as an
exhaustive data-handling pipeline. 

The classes in the data module help load data from HDF5 files that are
formatted in a particular way.  The HDF5 file must be structured with the
groups 'train','test','valid'.  Each of these groups must contain a group for
each batch, with the group name 'batchN' where N is an integer in the range 0
to the number of batches.  Each batch group must have datasets inside it that
contain the data of interest.

HDF5 Data Format
----------------

HDF5 data files begin with a top-level group with the name '/'.  A supported
HDF5 file for MNIST might have a group structure and data as follows (**Note:**
data field names 'pixels' and 'number' are just examples, and can be
arbitrarily named):

- /

  - test

    - batch0

      - pixels: :class:`numpy.ndarray` of pixels with shape **(batchsize, channels, width, height)**

      - number: :class:`numpy.ndarray` of labels with shape **(batchsize, 10)** containing a one-hot encoding of the number shown in each image

    - batch1

      - pixels

      - number

    - ...

    - batchN

  - train

    - batch0

      - pixels

      - number

    - batch1

      - pixels

      - number

    - ...

    - batchM

Reading HDF5 Data
-----------------

Data from the hypothetical HDF5 file for MNIST described above could be read using the classes in the :mod:`xnn.data` module as follows.

First, :class:`xnn.data.HDF5FieldReader` objects would be created to extract information about 'pixels' and 'number' from the HDF5 file.  These  :class:`xnn.data.HDF5FieldReader` objects allow specification of preprocessing functions, for example flattening images from the 'pixels' dataset into vectors, or conversion of 'number' labels from one-hot coding to integer coding.

Next, the :class:`xnn.data.HDF5FieldReader` objects are given to a :class:`xnn.data.HDF5BatchLoad` object, which is responsible for reading all the required data from a single batch group from the HDF5 file.

The :class:`xnn.data.HDF5BatchLoad` can then be given to a :class:`xnn.data.HDF5RamPool` which loads data from the HDF5 file into RAM, and returns a pool of these examples.

The pool returned by the :class:`xnn.data.HDF5RamPool` can be given to a :class:`xnn.data.samplers.Sampler`, which will select elements from the pool to return as batches for use with training or evaluating a network.

Data Package
============

xnn.data.HDF5BatchLoad module
-----------------------------

.. automodule:: xnn.data.HDF5BatchLoad
    :members:
    :undoc-members:
    :show-inheritance:

xnn.data.HDF5FieldReader module
-------------------------------

.. automodule:: xnn.data.HDF5FieldReader
    :members:
    :undoc-members:
    :show-inheritance:

xnn.data.HDF5Preprocessors module
---------------------------------

.. automodule:: xnn.data.HDF5Preprocessors
    :members:
    :undoc-members:
    :show-inheritance:

xnn.data.HDF5RamPool module
---------------------------

.. automodule:: xnn.data.HDF5RamPool
    :members:
    :undoc-members:
    :show-inheritance:

xnn.data.samplers module
------------------------

.. automodule:: xnn.data.samplers
    :members:
    :undoc-members:
    :show-inheritance:

xnn.data.weights module
-----------------------

.. automodule:: xnn.data.weights
    :members:
    :undoc-members:
    :show-inheritance:


