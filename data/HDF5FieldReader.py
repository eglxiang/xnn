import h5py
import numpy as np
import theano
from HDF5Preprocessors import *
from copy import deepcopy

class HDF5FieldReader(object):
    def __init__(self, name, fields=None, preprocessFunc=None, missingValue=None):
        """
        Read data from the HDF5 batch
        :param name: What name to associate with read and preprocessed data
        :param fields: list of fields to read.  if a element i is a str, read batch[field[i]].  if element i is a tuple, read batch[field[i][0]][field[i][1]]
        :param preprocessFunc: function to apply to data before it is returned.  This function will get a list of numpy arrays, and return a single numpy array
        :param missingValueFlag: If not none, labels matching this float value will be set to NaN
        """
        self.name=name
        if fields is None:
            self.fields=name
        else:
            self.fields=fields
        self.preprocessFunc=preprocessFunc
        self.missingValue=missingValue

    def getName(self):
        return self.name

    def readBatch(self,batch):
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
        properties = deepcopy(self.__dict__) 
        for k in properties:
            if hasattr(properties[k],'to_dict'):
                properties[k] = properties[k].to_dict()
        return properties


    

pixelReader =HDF5FieldReader('pixels',['pixels'],preprocessFunc=pixelPreprocess())
genderReader=HDF5FieldReader('isMale',['isMale'],missingValue=-12345.)
ageReader = HDF5FieldReader('age',['age'],missingValue=-12345.)
ageGuessSoftReader = HDF5FieldReader('age_guesses_soft',['age_guesses'],missingValue=-12345.,preprocessFunc=ageG_to_soft())
ageGuessHardReader = HDF5FieldReader('age_guesses_hard',['age_guesses'],missingValue=-12345.,preprocessFunc=ageG_to_hard())
ethnicityReader = HDF5FieldReader('ethnicitylabels',[('ethnicitylabels','asian'),('ethnicitylabels','black'),('ethnicitylabels','hispanic'),('ethnicitylabels','indian'),('ethnicitylabels','white')],preprocessFunc=np.hstack,missingValue=-12345.) 
impathReader = HDF5FieldReader('imagepaths',['imagepath'])
pixelhashReader=HDF5FieldReader('pixelhashid',['pixelhashid'])
identityReader = HDF5FieldReader('identity',['personidentity'])

dgfxReaderList = [pixelReader,genderReader,ageReader,ageGuessSoftReader,ageGuessHardReader,ethnicityReader,impathReader,pixelhashReader,identityReader]

