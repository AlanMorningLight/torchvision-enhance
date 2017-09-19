from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
from skimage.external import tifffile
from skimage import transform, filters
from skimage import util
from skimage.util.dtype import img_as_uint,img_as_ubyte
import cv2

"""
# 思路：
0. transform中所有输入的img都应该是np.ndarray or torch.tenor
1. 通用方法写成函数，其它的类调用这些函数，这样类和函数就可以同时使用
2. 如果是分类问题，只需要处理input，那么直接像原来一样使用compose调用相应的class或者lamdba
3. 如果是分割问题，需要同时操作input和target，那么user自己编写函数，生成相应的param去操作
# support operation
resize
center_crop
random_crop
flip
horizontal_flip
vertical_flip
rotate
shift
normaize
noise
dropout
"""


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _get_dtype_max(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 65535


def to_tensor(pic):
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        #get max value of dtype
        denominator = _get_dtype_max(pic)
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(denominator)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def to_ndarray(pic):
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    mode = None
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
    assert isinstance(npimg, np.ndarray)
    if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]

        if npimg.dtype == np.uint8:
            mode = 'L'
        if npimg.dtype == np.int16:
            mode = 'I;16'
        if npimg.dtype == np.int32:
            mode = 'I'
        elif npimg.dtype == np.float32:
            mode = 'F'
    else:
        if npimg.dtype == np.uint8:
            mode = 'RGB'
    assert mode is not None, '{} is not supported'.format(npimg.dtype)
    return npimg
    # return Image.fromarray(npimg, mode=mode)


def normalize(tensor, mean, std):
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def flip(img, flip_mode):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not (isinstance(flip_mode, int)):
        raise TypeError('flipCode should be integer. Got {}'.format(type(flip_mode)))
    return cv2.flip(img, flip_mode)

class Flip(object):
    """
    flip the input ndarray image
    three case:
        1. left to right
        2. top to bottom
        2. left to right and then top to bottom
    """

    def __init__(self, flip_mode):
        assert isinstance(flip_mode, int)
        self.flip_mode = flip_mode
    
    def __call__(self, img):
        
        return flip(img, self.flip_mode)

class RandomFlip(object):
    """
    random flip the ndarray image
    random probability (default init):
        1. original: 0.25
        2. left to right: 0.25
        3. top to bottom: 0.25
        4. left to right and then top to bottom: 0.25
    """

    # def __init__(self, u=0.25):
    #     self.u = 0.25
    def __call__(self, img):
        flip_mode = random.randint(-1, 2)
        if flip_mode == 2:
            return img
        else:
            return flip(img, flip_mode)

def rotate(img, angle=0,order=1):
    """Rotate image by a certain angle around its center.

        Parameters
        ----------
        img : ndarray
            Input image.
        angle : integer
            Rotation angle in degrees in counter-clockwise direction.
        order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.

        Returns
        -------
        rotated : ndarray
                Rotated version of the input.

        Examples
        --------
        rotate(image, 30)
        rotate(image, 180)
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not (isinstance(angle, int)):
        raise TypeError('Angle should be integer. Got {}'.format(type(angle)))

    type = img.dtype
    img_new = transform.rotate(img, angle, order=order, preserve_range=True)
    img_new = img_new.astype(type)

    return img_new

class RandomRotate(object):
    """rotate the image.

        Args:
            probability:the probability of the operation(0<=double<=1)
        Example:
            >>> transform_multi.Randomrotate(probability=0.9)

    """

    def __init__(self, probability=0.5):

        if not 0 <= probability <= 1:
            raise ValueError('Randomrotate.probability error')
        self.probability = probability

    def __call__(self, img):
        angle = random.randint(0, 360)
        r = round(random.uniform(0, 1), 1)
        if r < self.probability:
            return rotate(img, angle)
        else:
            return img

def shift(img, rightshift=5, downshift=5):
    """
    Args:
    img: the image input
    rightshift: the pixels of shift right
    downshift: the pixels of down right

    :return
    transformed img

    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not (isinstance(rightshift, int)):
        raise TypeError('shift.rightshift should be integer. Got {}'.format(type(rightshift)))
    if not (isinstance(downshift, int)):
        raise TypeError('shift.downshift should be integer. Got {}'.format(type(downshift)))

    type = img.dtype
    tform = transform.SimilarityTransform(translation=(-rightshift, -downshift))
    img_new = transform.warp(img, tform, preserve_range=True)
    img_new = img_new.astype(type)
    return img_new

class RandomShift(object):
    """shift the image.

        Args:
            probability:the probability of the operation (0<=double<=1)
            rightshift: the scale of the shift right (pixels)
            downshift:the scale of the shift down (pixels)
        Example:
            >>> transform_multi.RandomShift(probability=1, rightshift=10, downshift=10)

    """

    def __init__(self, probability=0.5, rightshift=5, downshift=5):
        if not 0 <= probability <= 1 :
            raise ValueError("RandomShift.probability is error")
        if not isinstance(rightshift,int):
            raise ValueError("RandomShift.rightshift is error")
        if not isinstance(downshift,int):
            raise ValueError("RandomShift.downshift is error")

        self.probability = probability
        self.rightshift = rightshift
        self.downshift = downshift

    def __call__(self, img):
        r = round(random.uniform(0,1),1)
        # print(r, self.probability, self.rightshift, self.downshift)
        if r < self.probability:
            rightshift=random.randint(0,self.rightshift)
            downshift = random.randint(0, self.downshift)
            return shift(img, rightshift=rightshift, downshift=downshift)
        else:
            return img

def randomcrop(img, outsize=(224,224)):
    """Crop a image at the same time

        Parameters
        ----------
        img : ndarray
        outsize : the shape of the croped image

        Returns
        -------
        Croped_image : ndarray

        Examples
        --------
        img_new = Crop(img, outsize=(256,256))

    """

    type = img.dtype
    if outsize[0] > img.shape[0] or outsize[1] > img.shape[1]:
        raise ValueError("SegRandomCrop.outsize larger than the input image")
    src_rows = img.shape[0]
    src_cols = img.shape[1]
    out_rows = outsize[0]
    out_cols = outsize[1]

    random_rows_up = np.random.randint(0, int(src_rows - out_rows + 1))
    random_cols_left = np.random.randint(0, int(src_cols - out_cols + 1))
    random_rows_down = src_rows - out_rows - random_rows_up
    random_cols_right = src_cols - out_cols - random_cols_left

    img_new = util.crop(img, ((random_rows_up, random_rows_down), (random_cols_left, random_cols_right), (0, 0)))

    img_new = img_new.astype(type)
    return img_new

def crop(img,
         random_rows_up=0,
         random_rows_down=0,
         random_cols_left=0,
         random_cols_right=0,
         deep=3
         ):
    """Crop a image at the same time

        Parameters
        ----------
        img : input
        random_rows_up:Number of values to remove from the edges in the top.
        random_rows_down:Number of values to remove from the edges in the bottom.
        random_cols_left:Number of values to remove from the edges in the left
        random_cols_right:Number of values to remove from the edges in the right
        deep: the dimentions of the input image(1 or 3)
              1: one dimention image such as (256,256)
              3: multi dimentions image such as (256,256,3) or (256,256,7)
        Returns
        -------
        Croped_image : ndarray

    """

    type = img.dtype
    random_rows_up = random_rows_up
    random_cols_left = random_cols_left
    random_rows_down = random_rows_down
    random_cols_right = random_cols_right

    if deep==1:
        img_new = util.crop(img, ((random_rows_up, random_rows_down), (random_cols_left, random_cols_right)))
        #img_new = util.crop(((first_dimention),(sec_dimen),(third_dimen)))
    elif deep==3:
        img_new = util.crop(img, ((random_rows_up, random_rows_down), (random_cols_left, random_cols_right), (0, 0)))
        # img_new = util.crop(((first_dimention),(sec_dimen),(third_dimen)))

    img_new = img_new.astype(type)
    return img_new

class RandomCrop(object):
    """randomcrop the image.

        Args:
            outsize: the shape of the croped image
        Example:
            >>> transform_multi.RandomCrop((256,256))

    """
    def __init__(self,outsize=(224,224)):
        self.outsize = outsize

    def __call__(self, img):
        if self.outsize[0]>img.shape[0] or self.outsize[1]>img.shape[1]:
            raise ValueError("RandomCrop.outsize larger than the input image")

        return randomcrop(img, self.outsize)

def noise(img, uintpara=16, Mean=0, Var=None):
    """add gaussian noise to the image

        Parameters
        ----------
        img : ndarray
        uintpara: the deep of the image(8 or 16)
                  8: uint8
                  16: uint16
        Mean:mean of the noise(generally 0)
        Var:variance of the noise
            0.01: generally suitable for uint8 image
            0.00001: generally suitable for uint16 image

        Returns
        -------
        Noised_image : ndarray
    """

    type = img.dtype


    if Var==None and type=='uint8':
        Var = 0.01
    if Var==None and type=='uint16':
        Var = 0.00001
    if Var == None and uintpara==8:
        Var = 0.01
    if Var == None and uintpara==16:
        Var = 0.00001

    if uintpara==8:
        img = img.astype(np.uint8)
        img_new = util.random_noise(img, mode='gaussian', mean=Mean, var=Var)
        img_new = img_as_ubyte(img_new)
    if uintpara==16:
        img = img.astype(np.uint16)
        img_new = util.random_noise(img, mode='gaussian', mean=Mean, var=Var)
        img_new = img_as_uint(img_new)

    img_new = img_new.astype(type)

    return img_new

class RandomNoise(object):
    """add gaussian noise to the image

            Parameters
            ----------
            probability : the probability of the operation (0<=double<=1)
            uintpara: the deep of the image(8 or 16)
                      8: uint8
                      16: uint16
            Mean:mean of the noise(generally 0)
            Var:variance of the noise
                0.01: generally suitable for uint8 image
                0.00001: generally suitable for uint16 image

            Returns
            -------
            Noised_image : ndarray
    """


    def __init__(self,probability, uintpara=16, mean=0, var=None):
        if not 0<= probability <=1 :
            raise ValueError('AddNoise.probability error')
        if not (uintpara ==8 or uintpara==16):
            raise ValueError('RandomNoise.uintpara error')

        self.uintpara=uintpara
        self.probability = probability
        self.mean = mean
        self.var = var

    def __call__(self, img):
        r = round(random.uniform(0, 1), 1)
        print(r, self.probability, self.mean, self.var)
        if r < self.probability:
            return noise(img, self.uintpara, self.mean, self.var)
        else:
            return img

def gaussianblur(img, sigma=1, multichannel=True):
    """Multi-dimensional Gaussian filter.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or color) to filter.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    multichannel : bool, optional (default: None)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together). Only 3 channels are supported. If `None`,
        the function will attempt to guess this, and raise a warning if
        ambiguous, when the array has shape (M, N, 3).

    Returns
    -------
    filtered_image : ndarray
    """

    type = img.dtype
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    img_new = filters.gaussian(img, sigma, multichannel, preserve_range=True)
    img_new = img_new.astype(type)
    return img_new

class GaussianBlur(object):
    """add gaussian noise to the image

        Parameters
        ----------
        probability : the probability of the operation (0<=double<=1)
        sigma :　　　 scalar or sequence of scalars, optional
                    Standard deviation for Gaussian kernel. The standard
                    deviations of the Gaussian filter are given for each axis as a
                    sequence, or as a single number, in which case it is equal for
                    all axes.
    　　multichannel : bool, optional (default: None)
                    Whether the last axis of the image is to be interpreted as multiple
                    channels. If True, each channel is filtered separately (channels are
                    not mixed together). Only 3 channels are supported. If `None`,
                    the function will attempt to guess this, and raise a warning if
                    ambiguous, when the array has shape (M, N, 3).

        Returns
        -------
        blured_image : ndarray
    """

    def __init__(self, probability, sigma=1, multichannel=True):
        if not 0<= probability <=1 :
            raise ValueError('GaussianBlur.probability error')
        if sigma<0:
            raise ValueError('GaussianBlur.sigma error')
        self.probability = probability
        self.sigma = sigma
        self.multichannel=multichannel

    def __call__(self, img):
        r = round(random.uniform(0, 1), 1)
        print(r, self.probability)
        if r < self.probability:
            return gaussianblur(img, self.sigma, self.multichannel)
        else:
            return img

def piecetransform(image, numcols=5, numrows=5, warp_left_right=10, warp_up_down=10,order=1):
    """2D piecewise affine transformation.

        Control points are used to define the mapping. The transform is based on
        a Delaunay triangulation of the points to form a mesh. Each triangle is
        used to find a local affine transform.

        Parameters
        ----------
        img : ndarray
        numcols : int, optional (default: 5)
            numbers of the colums to transformation
        numrows : int, optional (default: 5)
            numbers of the rows to transformation
        warp_left_right: int, optional (default: 10)
            the pixels of transformation left and right
        warp_up_down: int, optional (default: 10)
            the pixels of transformation up and down
        Returns
        -------
        Transformed_image : ndarray

        Examples
        --------
            >>> Transformed_img = piecetransform(image,numcols=10, numrows=10, warp_left_right=5, warp_up_down=5)

    """

    type = image.dtype

    rows, cols = image.shape[0], image.shape[1]

    numcols = numcols
    numrows = numrows

    src_cols = np.linspace(0, cols, numcols, dtype=int)
    src_rows = np.linspace(0, rows, numrows, dtype=int)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    src_rows_new = np.ndarray.transpose(src_rows)
    src_cols_new = np.ndarray.transpose(src_cols)
    # src_new = np.dstack([src_cols_new.flat, src_rows_new.flat])[0]

    dst_cols = np.ndarray(src_cols.shape)
    dst_rows = np.ndarray(src_rows.shape)
    for i in range(0, numcols):
        for j in range(0, numrows):
            if src_cols[i, j] == 0 or src_cols[i, j] == cols:
                dst_cols[i, j] = src_cols[i, j]
            else:
                dst_cols[i, j] = src_cols[i, j] + np.random.uniform(-1, 1) * warp_left_right

            if src_rows[i, j] == 0 or src_rows[i, j] == rows:
                dst_rows[i, j] = src_rows[i, j]
            else:
                dst_rows[i, j] = src_rows[i, j] + np.random.uniform(-1, 1) * warp_up_down

    dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]

    # dst_rows_new = np.ndarray.transpose(dst_rows)
    # dst_cols_new = np.ndarray.transpose(dst_cols)
    # dst_new = np.dstack([dst_cols_new.flat, dst_rows_new.flat])[0]

    out_rows = rows
    out_cols = cols

    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    img_new = transform.warp(image, tform, output_shape=(out_rows, out_cols), order=order, preserve_range=True)

    img_new = img_new.astype(type)
    return img_new


class PieceTransfor(object):
    """add gaussian noise to the image

        Parameters
        ----------
        probability : the probability of the operation (0<=double<=1)
        numcols : int, optional (default: 5)
            numbers of the colums to transformation
        numrows : int, optional (default: 5)
            numbers of the rows to transformation
        warp_left_right: int, optional (default: 10)
            the pixels of transformation left and right
        warp_up_down: int, optional (default: 10)
            the pixels of transformation up and down

        Returns
        -------
        Transformed_image : ndarray
    """

    def __init__(self, probability, numcols=10, numrows=10, warp_left_right=10, warp_up_down=10):
        if not 0<= probability <=1 :
            raise ValueError('PieceTransfor.probability error')
        if numcols<0 :
            raise ValueError('PieceTransfor.numcols error')
        if numrows<0 :
            raise ValueError('PieceTransfor.numrows error')
        if warp_left_right<0 :
            raise ValueError('PieceTransfor.warp_left_right error')
        if warp_up_down<0 :
            raise ValueError('PieceTransfor.warp_up_down error')

        self.probability = probability
        self.numcols = numcols
        self.numrows = numrows
        self.warp_left_right = warp_left_right
        self.warp_up_down = warp_up_down

    def __call__(self, img):
        r = round(random.uniform(0, 1), 1)
        print(r, self.probability)
        if r < self.probability:
            return piecetransform(img, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
        else:
            return img

def centercrop(img,outsize,target=None):
    """
        Parameters
        ----------
        img : ndarray
        outsize : the size of out image
        target : is or not the targrt(uesed for the semantic segmentation)

        Returns
        -------
        Croped_image : ndarray

        Examples
        --------
        >>> Transformed_img = centercrop(image, [224,224])

    """

    if outsize[0] > img.shape[0] or outsize[1] > img.shape[1]:
        raise ValueError("centercrop.outsize larger than the input image")
    type = img.dtype
    src_rows = img.shape[0]
    src_cols = img.shape[1]
    out_rows = outsize[0]
    out_cols = outsize[1]

    random_rows_up = int((src_rows - out_rows) / 2.)
    random_cols_left = int((src_cols - out_cols) / 2.)
    random_rows_down = src_rows - out_rows - random_rows_up
    random_cols_right = src_cols - out_cols - random_cols_left

    if target==None:
       img_new = util.crop(img, ((random_rows_up, random_rows_down), (random_cols_left, random_cols_right), (0, 0)))
       img_new = img_new.astype(type)
       return img_new
    elif target==True:
        img_new = util.crop(img, ((random_rows_up, random_rows_down), (random_cols_left, random_cols_right)))
        img_new = img_new.astype(type)
        return img_new

class CenterCrop(object):
    """Crops the given Image at the center.

    Args:
        outsize : the size of out image
        Examples
        --------
        >>> Transformed_img = CenterCrop([224,224])
    """

    def __init__(self, outsize):
        self.outsize = outsize

    def __call__(self, img):
        """
        Args:
            img : Image to be cropped.

        Returns:
            Cropped image.
        """
        if self.outsize[0] > img.shape[0] or self.outsize[1] > img.shape[1]:
            raise ValueError("CenterCrop.outsize larger than the input image")

        type = img.dtype
        src_rows = img.shape[0]
        src_cols = img.shape[1]
        out_rows = self.outsize[0]
        out_cols = self.outsize[1]

        random_rows_up = int((src_rows - out_rows)/2.)
        random_cols_left = int((src_cols - out_cols)/2.)
        random_rows_down = src_rows - out_rows - random_rows_up
        random_cols_right = src_cols - out_cols - random_cols_left

        img_new = util.crop(img, ((random_rows_up, random_rows_down), (random_cols_left, random_cols_right), (0, 0)))
        img_new = img_new.astype(type)
        return img_new

def pad(img,pad_width=[(0,0),(0,0),(0,0)], mode='reflect',**kwargs):
    """Pad the given image.

    Args:
        pad_width : {sequence, array_like, int}
                    Number of values padded to the edges of
                    each axis. ((before_1, after_1), ...
                    (before_N, after_N)) unique pad widths
                     for each axis. ((before, after),) yields
                     same before and after pad for each axis.
                    (pad,) or int is a shortcut for before = after
                    = pad width for all axes.
        mode: str or function. contain{‘constant’,‘edge’,‘linear_ramp’,‘maximum’,‘mean’
            , ‘median’, ‘minimum’, ‘reflect’,‘symmetric’,‘wrap’}
    Examples
        --------
        >>> Transformed_img = pad(img,[(20,20),(20,20),(0,0)],mode='reflect')
    """

    pad_width = pad_width
    mode = mode
    return np.pad(img, pad_width, mode)

class Pad(object):
    """Pad the given image.

    Args:
        pad_width : {sequence, array_like, int}
                    Number of values padded to the edges of
                    each axis. ((before_1, after_1), ...
                    (before_N, after_N)) unique pad widths
                     for each axis. ((before, after),) yields
                     same before and after pad for each axis.
                    (pad,) or int is a shortcut for before = after
                    = pad width for all axes.
        mode: str or function. contain{‘constant’,‘edge’,‘linear_ramp’,‘maximum’,‘mean’
            , ‘median’, ‘minimum’, ‘reflect’,‘symmetric’,‘wrap’}
    Examples
        --------
        >>> paded=Pad([(10,10),(20,20),(0,0)],"reflect")

    """

    def __init__(self, pad_width=[(0,0),(0,0),(0,0)], mode='reflect',**kwargs):

        self.pad_width = pad_width
        self.mode = mode

    def __call__(self, img):
        return np.pad(img, self.pad_width, self.mode)

def resize(img, outsize=None, interpolation=cv2.INTER_NEAREST):
    """resize the image.

        Args:
            img : the input image
            outsize : the shape of the out image
            interpolation: the model of the  interpolation{}

        Examples
            --------
            >>> resized=resize(img,[(200,300)])
    """


    if outsize==None:
       outsize1=img.shape[0]
       outsize2 = img.shape[1]
    else:
        outsize1 = outsize[0]
        outsize2 = outsize[1]

    return cv2.resize(img, (outsize1,outsize2),interpolation)

class Resize(object):
    """resize the image.

        Args:
            img : the input image
            outsize : the shape of the out image
            interpolation: the model of the  interpolation
                   INTER_NEAREST - a nearest-neighbor interpolation
                   INTER_LINEAR - a bilinear interpolation (used by default)
                   INTER_AREA - resampling using pixel area relation. It may
                                be a preferred method for image decimation, as it gives moire’-free results.
                                But when the image is zoomed, it is similar to the INTER_NEAREST method.
                   INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
                   INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

        Examples
            --------
            >>> resized=Resize([(200,300)])
    """

    def __init__(self, outsize,interpolation=cv2.INTER_NEAREST):
        self.outsize = outsize
        self.interpolation=interpolation
    def __call__(self, img):
        img = img.astype(dtype=np.float)
        if self.outsize[0] > img.shape[0] or self.outsize[1] > img.shape[1]:
            raise ValueError("Resize.outsize larger than the input image")

        img_resize=cv2.resize(img,(self.outsize[0],self.outsize[1]),self.interpolation)
        img_resize = img_resize.astype(dtype=np.int32)
        print(img_resize)
        return img_resize

##======================= semantic segmentation =============================================
class SegRandomRotate(object):
    """rotate the image.

        Args:
            probability:the probability of the operation(0<=double<=1)
        Example:
            >>> transform_multi.Randomrotate(probability=0.9)

    """


    def __init__(self, probability=0.5):

        if not 0 <= probability <= 1:
            raise ValueError('SegRandomRotate.probability error')
        self.probability = probability

    def __call__(self, img, target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegRandomRotate.img.size not match the SegRandomRotate.target.size")
        # if target==None:
        #     raise ValueError('SegRandomRotate has no target parameters ')
        angle = random.randint(0, 360)
        r = round(random.uniform(0, 1), 1)
        if r < self.probability:
            img_trans= rotate(img, angle)
            target_trans=rotate(target, angle, order=0)
            return img_trans, target_trans
        else:
            return img, target

class SegRandomShift(object):
    """shift the image.

            Args:
                probability:the probability of the operation (0<=double<=1)
                rightshift: the scale of the shift right (pixels)
                downshift:the scale of the shift down (pixels)
            Example:
                >>> transform_multi.RandomShift(probability=1, rightshift=10, downshift=10)

    """

    def __init__(self, probability=0.5, rightshift=5, downshift=5):
        if not 0 <= probability <= 1:
            raise ValueError("SegRandomShift.probability is error")
        if not isinstance(rightshift, int):
            raise ValueError("SegRandomShift.rightshift is error")
        if not isinstance(downshift, int):
            raise ValueError("SegRandomShift.downshift is error")

        self.probability = probability
        self.rightshift = rightshift
        self.downshift = downshift

    def __call__(self, img,target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegRandomShift.img.size not match the SegRandomShift.target.size")
        # if target==None:
        #     raise ValueError('SegRandomShift has no target parameters ')
        r = round(random.uniform(0, 1), 1)
        if r < self.probability:
            rightshift = random.randint(0, self.rightshift)
            downshift = random.randint(0, self.downshift)
            img_trans = shift(img, rightshift=rightshift, downshift=downshift)
            target_trans =shift(target, rightshift=rightshift, downshift=downshift)
            return img_trans, target_trans
        else:
            return img, target

class SegRandomCrop(object):
    """randomcrop the image.

            Args:
                outsize: the shape of the croped image
            Example:
                >>> transform_multi.RandomCrop((256,256))

    """

    def __init__(self, outsize=(224, 224)):
        self.outsize = outsize

    def __call__(self, img, target):

        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegRandomCrop.img.size not match the SegRandomCrop.target.size")
        # if target==None:
        #     raise ValueError('SegRandomCrop has no target parameters ')
        if self.outsize[0] > img.shape[0] or self.outsize[1] > img.shape[1]:
            raise ValueError("SegRandomCrop.outsize larger than the input image")
        src_rows = img.shape[0]
        src_cols = img.shape[1]
        out_rows = self.outsize[0]
        out_cols = self.outsize[1]

        random_rows_up = np.random.randint(0, int(src_rows - out_rows + 1))
        random_cols_left = np.random.randint(0, int(src_cols - out_cols + 1))
        random_rows_down = src_rows - out_rows - random_rows_up
        random_cols_right = src_cols - out_cols - random_cols_left

        img_trans = crop(img, random_rows_up,random_rows_down, random_cols_left, random_cols_right,deep=3)
        target_trans = crop(target, random_rows_up,random_rows_down, random_cols_left, random_cols_right,deep=1)
        return img_trans, target_trans

class SegRandomNoise(object):
    """add gaussian noise to the image

        Parameters
        ----------
        probability : the probability of the operation (0<=double<=1)
        uintpara: the deep of the image(8 or 16)
                  8: uint8
                  16: uint16
        Mean:mean of the noise(generally 0)
        Var:variance of the noise
            0.01: generally suitable for uint8 image
            0.00001: generally suitable for uint16 image

        Returns
        -------
        Noised_image : ndarray
    """

    def __init__(self, probability, uintpara=16, mean=0, var=None):
        if not 0 <= probability <= 1:
            raise ValueError('SegRandomNoise.probability error')
        if not (uintpara == 8 or uintpara == 16):
            raise ValueError('SegRandomNoise.uintpara error')

        self.uintpara = uintpara
        self.probability = probability
        self.mean = mean
        self.var = var

    def __call__(self, img, target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegRandomNoise.img.size not match the SegRandomNoise.target.size")
        # if target==None:
        #     raise ValueError('SegRandomNoise has no target parameters ')
        r = round(random.uniform(0, 1), 1)
        # print(r, self.probability, self.mean, self.var)
        if r < self.probability:
            img_trans = noise(img, self.uintpara, self.mean, self.var)

            return img_trans, target
        else:
            return img, target

class SegGaussianBlur(object):
    """add gaussian noise to the image

            Parameters
            ----------
            probability : the probability of the operation (0<=double<=1)
            sigma :　　　 scalar or sequence of scalars, optional
                        Standard deviation for Gaussian kernel. The standard
                        deviations of the Gaussian filter are given for each axis as a
                        sequence, or as a single number, in which case it is equal for
                        all axes.
        　　multichannel : bool, optional (default: None)
                        Whether the last axis of the image is to be interpreted as multiple
                        channels. If True, each channel is filtered separately (channels are
                        not mixed together). Only 3 channels are supported. If `None`,
                        the function will attempt to guess this, and raise a warning if
                        ambiguous, when the array has shape (M, N, 3).

            Returns
            -------
            blured_image : ndarray
    """

    def __init__(self, probability, sigma=1, multichannel=True):
        if not 0 <= probability <= 1:
            raise ValueError('SegGaussianBlur.probability error')
        if sigma < 0:
            raise ValueError('SegGaussianBlur.sigma error')
        self.probability = probability
        self.sigma = sigma
        self.multichannel = multichannel

    def __call__(self, img, target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegGaussianBlur.img.size not match the SegGaussianBlur.target.size")
        # if target==None:
        #     raise ValueError('SegGaussianBlur has no target parameters ')
        r = round(random.uniform(0, 1), 1)
        # print(r, self.probability)
        if r < self.probability:
            img_trans = gaussianblur(img, self.sigma, self.multichannel)
            return img_trans, target
        else:
            return img, target

class SegPieceTransfor(object):
    """add gaussian noise to the image

            Parameters
            ----------
            probability : the probability of the operation (0<=double<=1)
            numcols : int, optional (default: 5)
                numbers of the colums to transformation
            numrows : int, optional (default: 5)
                numbers of the rows to transformation
            warp_left_right: int, optional (default: 10)
                the pixels of transformation left and right
            warp_up_down: int, optional (default: 10)
                the pixels of transformation up and down

            Returns
            -------
            Transformed_image : ndarray
    """

    def __init__(self, probability, numcols=10, numrows=10, warp_left_right=10, warp_up_down=10):
        if not 0 <= probability <= 1:
            raise ValueError('SegPieceTransfor.probability error')
        if numcols < 0:
            raise ValueError('SegPieceTransfor.numcols error')
        if numrows < 0:
            raise ValueError('SegPieceTransfor.numrows error')
        if warp_left_right < 0:
            raise ValueError('SegPieceTransfor.warp_left_right error')
        if warp_up_down < 0:
            raise ValueError('SegPieceTransfor.warp_up_down error')

        self.probability = probability
        self.numcols = numcols
        self.numrows = numrows
        self.warp_left_right = warp_left_right
        self.warp_up_down = warp_up_down

    def __call__(self, img, target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegPieceTransfor.img.size not match the SegPieceTransfor.target.size")
        # if target==None:
        #     raise ValueError('SegPieceTransfor has no target parameters ')
        r = round(random.uniform(0, 1), 1)
        # print(r, self.probability)
        if r < self.probability:
            img_trans = piecetransform(img, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
            target_trans = piecetransform(target, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down,order=0)
            return img_trans,target_trans
        else:
            return img, target

def SegFlip(img, flip_mode):
    """
    flip the input ndarray image
    three case:
        1. left to right
        2. top to bottom
        2. left to right and then top to bottom
    """
    return flip(img, flip_mode)

class SegRandomFlip(object):
    """
    random flip the ndarray image
    random probability (default init):
        1. original: 0.25
        2. left to right: 0.25
        3. top to bottom: 0.25
        4. left to right and then top to bottom: 0.25
    """
    # def __init__(self):

    def __call__(self, img, target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegRandomFlip.img.size not match the SegRandomFlip.target.size")
        # if target==None:
        #     raise ValueError('SegRandomFlip has no target parameters ')
        flip_mode = random.randint(-1, 2)
        if flip_mode == 2:
            return img, target
        else:
            return flip(img, flip_mode), flip(target, flip_mode)

class SegPad(object):
    """Pad the given image.

    Args:
        pad_width : {sequence, array_like, int}
                    Number of values padded to the edges of
                    each axis. ((before_1, after_1), ...
                    (before_N, after_N)) unique pad widths
                     for each axis. ((before, after),) yields
                     same before and after pad for each axis.
                    (pad,) or int is a shortcut for before = after
                    = pad width for all axes.
        mode: str or function. contain{‘constant’,‘edge’,‘linear_ramp’,‘maximum’,‘mean’
            , ‘median’, ‘minimum’, ‘reflect’,‘symmetric’,‘wrap’}
    Examples
        --------
    """

    def __init__(self, pad_width=[(0, 0),(0, 0),(0, 0)], mode='reflect', **kwargs):

        self.img_pad_width = pad_width
        self.target_pad_width = pad_width[:2]
        self.mode = mode

    def __call__(self, img, target):
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegPad.img.size not match the SegPad.target.size")
        img_pad=np.pad(img, self.img_pad_width, self.mode)
        target_pad = np.pad(target, self.target_pad_width, self.mode)
        return img_pad, target_pad

class SegCenterCrop(object):
    """Crops the given Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, outsize):
        self.outsize = outsize

    def __call__(self, img, target):
        """
        Args:
            img : Image to be cropped.

        Returns:
            Cropped image.
        """
        if self.outsize[0] > img.shape[0] or self.outsize[1] > img.shape[1]:
            raise ValueError("CenterCrop.outsize larger than the input image")
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegCenterCrop.img.size not match the SegCenterCrop.target.size")

        type1=img.dtype
        type2=target.dtype
        src_rows = img.shape[0]
        src_cols = img.shape[1]
        out_rows = self.outsize[0]
        out_cols = self.outsize[1]

        random_rows_up = int((src_rows - out_rows) / 2.)
        random_cols_left = int((src_cols - out_cols) / 2.)
        random_rows_down = src_rows - out_rows - random_rows_up
        random_cols_right = src_cols - out_cols - random_cols_left

        img_new = util.crop(img,((random_rows_up, random_rows_down), (random_cols_left, random_cols_right), (0, 0)))
        img_new = img_new.astype(type1)

        target_new = util.crop(target,[(random_rows_up, random_rows_down), (random_cols_left, random_cols_right)])
        target_new = target_new.astype(type2)
        return img_new, target_new

class   SegResize(object):
    """resize the image.

        Args:
            img : the input image
            outsize : the shape of the out image
            interpolation: the model of the  interpolation
                   INTER_NEAREST - a nearest-neighbor interpolation
                   INTER_LINEAR - a bilinear interpolation (used by default)
                   INTER_AREA - resampling using pixel area relation. It may
                                be a preferred method for image decimation, as it gives moire’-free results.
                                But when the image is zoomed, it is similar to the INTER_NEAREST method.
                   INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
                   INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

        Examples
            --------
            >>> resized=Resize([(200,300)])
    """
    def __init__(self, outsize, interpolation=cv2.INTER_NEAREST):
        self.outsize = outsize
        self.interpolation = interpolation

    def __call__(self, img, target):
        if self.outsize[0] > img.shape[0] or self.outsize[1] > img.shape[1]:
            raise ValueError("SegResize.outsize larger than the input image")
        if not (img.shape[0]==target.shape[0] and img.shape[1]==target.shape[1]):
            raise ValueError("SegResize.img.size not match the SegResize.target.size")

        img = img.astype(dtype=np.float)
        target = target.astype(dtype=np.float)
        img_resize=cv2.resize(img, (self.outsize[0], self.outsize[1]), self.interpolation)
        target_resize=cv2.resize(target, (self.outsize[0], self.outsize[1]), interpolation=cv2.INTER_NEAREST)

        img_resize = img_resize.astype(dtype=np.int32)
        target_resize = target_resize.astype(dtype=np.int32)

        return img_resize, target_resize
# ====================================================================================================

# class Scale(object):
#     """Rescale the input PIL.Image to the given size.
#
#     Args:
#         size (sequence or int): Desired output size. If size is a sequence like
#             (w, h), output size will be matched to this. If size is an int,
#             smaller edge of the image will be matched to this number.
#             i.e, if height > width, then image will be rescaled to
#             (size * height / width, size)
#         interpolation (int, optional): Desired interpolation. Default is
#             ``PIL.Image.BILINEAR``
#     """
#
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
#         self.size = size
#         self.interpolation = interpolation
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL.Image): Image to be scaled.
#
#         Returns:
#             PIL.Image: Rescaled image.
#         """
#         return scale(img, self.size, self.interpolation)
#
# def scale(img, size, interpolation=Image.BILINEAR):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
#     if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
#         raise TypeError('Got inappropriate size arg: {}'.format(size))
#
#     if isinstance(size, int):
#         w, h = img.size
#         if (w <= h and w == size) or (h <= w and h == size):
#             return img
#         if w < h:
#             ow = size
#             oh = int(size * h / w)
#             return img.resize((ow, oh), interpolation)
#         else:
#             oh = size
#             ow = int(size * w / h)
#             return img.resize((ow, oh), interpolation)
#     else:
#         return img.resize(size, interpolation)


# def pad(img, padding, fill=0):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
#
#     if not isinstance(padding, (numbers.Number, tuple)):
#         raise TypeError('Got inappropriate padding arg')
#     if not isinstance(fill, (numbers.Number, str, tuple)):
#         raise TypeError('Got inappropriate fill arg')
#
#     if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
#         raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
#                          "{} element tuple".format(len(padding)))
#
#     return ImageOps.expand(img, border=padding, fill=fill)


# def crop(img, x, y, w, h):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
#
#     return img.crop((x, y, x + w, y + h))


def scaled_crop(img, x, y, w, h, size, interpolation=Image.BILINEAR):
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, x, y, w, h)
    img = scale(img, size, interpolation)


def hflip(img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class SegCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target



class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

class SegToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, target):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(img), target

# class ToPILImage(object):
#     """Convert a tensor to PIL Image.
#
#     Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
#     H x W x C to a PIL.Image while preserving the value range.
#     """
#
#     def __call__(self, pic):
#         """
#         Args:
#             pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.
#
#         Returns:
#             PIL.Image: Image converted to PIL.Image.
#
#         """
#         return to_pilimage(pic)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        return normalize(tensor, self.mean, self.std)


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        return scale(img, self.size, self.interpolation)

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class SegLambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, target):
        return self.lambd(img), target