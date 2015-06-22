import numpy as np
import theano
import scipy.ndimage.interpolation
from copy import deepcopy

def makeFloatX(labelslist):
    return labelslist[0].astype(theano.config.floatX)

class ageG_to_hard():
    def __init__(self, bins=[18,25,35,45,55,65,100]):
        self.bins = bins

    def __call__(self, labels):
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
        return OHlabels
    
    def to_dict(self):
        properties = deepcopy(self.__dict__)
        properties['funcName']='ageG_to_hard'
        return properties


class ageG_to_soft():
    # TODO: Clean this up
    def __init__(self, bins=[18,25,35,45,55,65,100]):
        self.bins = bins

    def __call__(self, labels):
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
        return labels
    
    def to_dict(self):
        properties = deepcopy(self.__dict__)
        properties['funcName']='ageG_to_soft'
        return properties
   
class pixelPreprocess():
    def __init__(self,roi=None,make_grayscale=False,flatten=True):
        self.roi = roi
        self.make_grayscale = make_grayscale 
        self.flatten = flatten

    def __call__(self, batchdata):
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
        properties = deepcopy(self.__dict__)
        properties['funcName']='pixelPreprocess'
        return properties

