import h5py
import numpy as np
import theano
from HDF5Preprocessors import *
from copy import deepcopy

class HDF5FieldReader(object):
    def __init__(self, name, fields=None, preprocessFunc=None, missingValue=None):
        """
        Read data from the HDF5 batch

        Parameters
        ----------

        name : str
            What name to associate with read and preprocessed data
        fields : list
            List of fields to read.  if element i is a str, read batch[fields[i]].  if element i is a tuple, read batch[fields[i][0]][fields[i][1]]
        preprocessFunc : function
            The function to apply to data before it is returned.  This function will get a list of numpy arrays, and return a single numpy array
        missingValueFlag : float or None 
            If not None, labels matching this float value will be set to NaN
        """
        self.name=name
        if fields is None:
            self.fields=name
        else:
            self.fields=fields
        self.preprocessFunc=preprocessFunc
        self.missingValue=missingValue

    def getName(self):
        """
        Gets the name of the this :class:`HDF5FieldReader`.

        Returns
        -------
        str
            The name of this :class:`HDF5FieldReader`.
        """
        return self.name

    def readBatch(self,batch):
        """
        Reads the fields for which this object is responsible from a batch of data.

        Parameters
        ----------

        batch : The HDF5 dataset that contains data from a single HDF5 batch.

        Returns
        -------
        :class:`numpy.ndarray`
            The preprocessed data from the fields for which this object is responsible.
        """
        if type(self.fields) is str:
            self.fields = [self.fields]
       
        vals = []
        for f in self.fields:
            if type(f) is str:
                value = batch[f].value
            else:
                value = batch
                for fsub in f:
                    value = value[fsub]
            if value.ndim == 1:
                value = value[:,np.newaxis]
            if self.missingValue is not None:
                value = value.astype(theano.config.floatX,copy=False) 
                value[value==self.missingValue] = np.nan
            vals.append(value)

        if self.preprocessFunc is None:
            if len(vals)>1:
                value = np.concatenate(vals,axis=1)
            else:
                value = vals[0]
        else:
            value = self.preprocessFunc(vals)

        return value

    def to_dict(self):
        """
        Serialize to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the :class:`HDF5FieldReader`
        """

        properties = deepcopy(self.__dict__) 
        for k in properties:
            if hasattr(properties[k],'to_dict'):
                properties[k] = properties[k].to_dict()
        return properties


    

pixelReader =HDF5FieldReader('pixels',['pixels'],preprocessFunc=pixelPreprocess())
genderReader=HDF5FieldReader('isMale',['isMale'],preprocessFunc=makeFloatX)
ageReader = HDF5FieldReader('age',['age'],preprocessFunc=makeFloatX)
ageGuessSoftReader = HDF5FieldReader('age_guesses_soft',['age_guesses'],preprocessFunc=ageG_to_soft())
ageGuessHardReader = HDF5FieldReader('age_guesses_hard',['age_guesses'],preprocessFunc=ageG_to_hard())
ethnicityReader = HDF5FieldReader('ethnicitylabels',[('ethnicitylabels','asian'),('ethnicitylabels','black'),('ethnicitylabels','hispanic'),('ethnicitylabels','indian'),('ethnicitylabels','white')],preprocessFunc=np.hstack) 
impathReader = HDF5FieldReader('imagepaths',['imagepath'])
pixelhashReader=HDF5FieldReader('pixelhashid',['pixelhashid'])
identityReader = HDF5FieldReader('identity',['personidentity'])

dgfxReaderList = [pixelReader,genderReader,ageReader,ageGuessSoftReader,ageGuessHardReader,ethnicityReader,impathReader,pixelhashReader,identityReader]

