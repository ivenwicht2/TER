from ImgTOnpy import extract,show_img
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt


def augment(image):
        transform = []
        for img in image :
            transform.append(rotate(img, angle=45, mode = 'wrap'))
        
        for img in image :
            transform.append(rotate(img, angle=130, mode = 'wrap'))
        
        transformation = AffineTransform(translation=(25,25))
        for img in image :
            transform.append(warp(img,transformation,mode='wrap'))

        for img in image :
            transform.append(np.fliplr(img))

        for img in image :
            transform.append(np.flipud(img))

        sigma=0.155
        for img in image :
            transform.append(random_noise(img,var=sigma**2))
        

        for tf in transform :
            image = np.concatenate((image,tf), axis=0)

        return image


if __name__ == '__main__':
    img,label = extract("DATA") 
    rotated = augment(img)[0]
    show_img([rotated,img[0]],["test","original"],2)