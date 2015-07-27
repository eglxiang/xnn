import numpy as np
import theano
import scipy.ndimage.interpolation
from copy import deepcopy

def makeFloatX(labelslist):
    """
    Converts data to theano's floatX type to help with running data on the GPU
    
    Parameters
    ----------
    labelslist : list
        list of labels

    Returns
    -------
    :class:`numpy.ndarray`
        numpy array converted to a floatX
    """
    return labelslist[0].astype(theano.config.floatX)

class ageG_to_hard(object):
    """
    Convert a length-100 histogram of age labels to a one-hot encoding.

    Parameters
    ----------

    bins : list
        right edges of categories for age binning
    """

    def __init__(self, bins=[18,25,35,45,55,65,100]):
        self.bins = bins

    def __call__(self, labels):
        """
        Convert labels into one-hot encoding

        Parameters
        ----------

        labels : list
            length 100 histogram of age labels

        Returns
        -------
        :class:`numpy.ndarray`
            one-hot representation of age groups

        """
        labels = labels[0]
        labs = np.zeros((labels.shape[0],len(self.bins)))
        prebe = 0
        for i,be in enumerate(self.bins):
            labs[:,[i]] += np.sum(labels[:,prebe:be],axis=1,keepdims=True)
            prebe = be
        with np.errstate(invalid='ignore'):
            dnm = np.nansum(labs,axis=1,keepdims=True)
            labels = labs / dnm

            mlab = np.max(labels,axis=1,keepdims=True)
            OHlabels = (labels==mlab).astype(theano.config.floatX)
            OHlabels[np.any(np.isnan(labels), axis=1), :] = np.nan

            morethanone = np.sum(OHlabels,axis=1,keepdims=True) > 1
            maxtoolow = mlab < 0#.5
            nanlabind = (morethanone|maxtoolow).flatten()
            OHlabels[nanlabind,:] = np.nan
        return OHlabels.astype(theano.config.floatX)
    
    def to_dict(self):
        """
        Return a dictionary representation of this preprocessing function.
        """
        properties = deepcopy(self.__dict__)
        properties['funcName']='ageG_to_hard'
        return properties


class ageG_to_soft(object):
    """
    Convert a length-100 histogram of age labels to a probability distribution
    over categories.

    Parameters
    ----------

    bins : list
        right edges of categories for age binning
    """
    # TODO: Clean this up
    def __init__(self, bins=[18,25,35,45,55,65,100]):
        self.bins = bins

    def __call__(self, labels):
        """
        Convert labels into probability distribution

        Parameters
        ----------

        labels : list
            length 100 histogram of age labels

        Returns
        -------
        :class:`numpy.ndarray`
            distribution of individual labeler responses across age groups

        """
        labels = labels[0]
        labs = np.zeros((labels.shape[0],len(self.bins)))
        prebe = 0
        for i,be in enumerate(self.bins):
            labs[:,[i]] += np.sum(labels[:,prebe:be],axis=1,keepdims=True)
            prebe = be
        with np.errstate(invalid='ignore'):
            labels = labs / np.nansum(labs,axis=1,keepdims=True)

            mlab = np.max(labels,axis=1,keepdims=True)
            OHlabels = (labels==mlab).astype(theano.config.floatX)

            morethanone = np.sum(OHlabels,axis=1,keepdims=True) > 1
            maxtoolow = mlab < 0#.5
            nanlabind = (morethanone|maxtoolow).flatten()
            OHlabels[nanlabind,:] = np.nan

            labels[nanlabind,:] = np.nan
        return labels.astype(theano.config.floatX)
    
    def to_dict(self):
        """
        Return a dictionary representation of this preprocessing function.
        """
        properties = deepcopy(self.__dict__)
        properties['funcName']='ageG_to_soft'
        return properties
   
class pixelPreprocess(object):
    """
    Process image data.
    
    Parameters
    ----------

    roi : dict
        A dictionary with keys 'x','y','width', and 'height' specifying a
        rectangle of pixels that should be extracted.

    make_grayscale : bool
        If true, RGB channels will be converted to grayscale.  Has no effect on
        grayscale images.

    flatten : bool
        If true, **(1, c, w, h)**-shaped images are vectorized to **(1,c*w*h)** 
    """
    def __init__(self,roi=None,make_grayscale=False,flatten=True):
        self.roi = roi
        self.make_grayscale = make_grayscale 
        self.flatten = flatten

    def __call__(self, batchdata):
        """
        Process image data.

        Parameters
        ----------
        batchdata : list
            list that contains image data in a batch

        Returns
        -------
        :class:`numpy.ndarray`
            batch of images that has been preprocessed.
        """
        pixels = batchdata[0]

        # crop to ROI if specified
        if self.roi:
            x1 = self.roi["x"]
            y1 = self.roi["y"]
            x2 = self.roi["x"]+self.roi["width"]
            y2 = self.roi["y"]+self.roi["height"]
            pixels = pixels[:, :, y1:y2, x1:x2]
            if "newshape" in self.roi.keys():
                pixelsnew = np.zeros((pixels.shape[0],pixels.shape[1],self.roi["newshape"][0],self.roi["newshape"][1]))
                zoomfactor = (1,self.roi["newshape"][0]/float(self.roi["width"]),self.roi["newshape"][1]/float(self.roi["height"]))
                for i in xrange(pixels.shape[0]):
                    pixelsnew[i,:,:,:] = scipy.ndimage.interpolation.zoom(pixels[i,:,:,:],zoomfactor,mode='nearest') 
                pixels = pixelsnew.astype(theano.config.floatX)
                #pixels = scipy.ndimage.interpolation.zoom(pixels,zoomfactor)

        # convert to grayscale
        # TODO: Consider using Opencv for grayscale to be consistent with runtime
        if (self.make_grayscale) and (pixels.shape[1] == 3):
            bgr = np.array([.114, .587, .299], dtype=theano.config.floatX)[np.newaxis, :, np.newaxis, np.newaxis]
            pixels = np.sum(pixels * bgr, axis=1, keepdims=True)

        # flatten images to a vector
        if self.flatten:
            pixels = pixels.reshape(-1, np.prod(pixels.shape[1:]))

        return pixels

    def to_dict(self):
        """
        Return a dictionary representation of this preprocessing function.
        """
        properties = deepcopy(self.__dict__)
        properties['funcName']='pixelPreprocess'
        return properties

