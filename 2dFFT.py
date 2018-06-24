from PIL import Image, ImageOps, ImageDraw
import numpy as np
from scipy import ndimage, fftpack
import cv2 as cv

#https://stackoverflow.com/questions/2598734/numpy-creating-a-complex-array-from-2-real-ones

def drawPoint(array, number):

    x = number / 4
    y = number % 4
    
    x = x + array.shape[0]/2 - 1
    y = y + array.shape[0]/2 - 1
    
    array[x,y] = 255

def hartley( n ):

  import numpy as np

  a = np.zeros ( ( n, n ) )

  for i in range ( 0, n ):
    for j in range ( 0, n ):

      angle = 2.0 * np.pi * float ( i * j ) / float ( n )

      a[i,j] = np.sin ( angle ) + np.cos ( angle )

  return a

for i in range(16 ** 3):

    print i

  
    #im = Image.open('blocky_2.png')

    range = np.linspace(.28,.04,50)

    #imArray = np.asarray(im)

    imArray = np.zeros((512,512,3))

    #editIm = Image.fromarray(np.uint8(imArray))

    #draw = ImageDraw.Draw(editIm)
    
    base = 16
    
    drawPoint(imArray, i % base)
    drawPoint(imArray, (i / base) % base)
    drawPoint(imArray, ((i / base) / base) % base)
    
    ifrac = i / 3.0


#imArray[400,400,0] = 0
#imArray[0:800,401:800,0] = 255
#imArray[390:400,401:410,0] = 255
#imArray[401:410,390:400,0] = 255
#imArray[390:400,401:420,1] = 255
#imArray[401:420,390:400,1] = 255
#imArray[395:400,401:405,2] = 255
#imArray[401:405,395:400,2] = 255
#imArray[350:500,401:420,:] = 0
#imArray[401,400,:] = 255
#imArray[381,360,:] = 255
#imArray[401,401,0] = 0
#imArray[401,402,:] = 0

    imArray = np.rint( imArray)

#imArray = np.asarray(im)

    im = Image.fromarray(np.uint8(imArray), mode = 'RGB')

    im.save('output/inputs/%s.png' %i)

    r = imArray[:,:,0]
    g = imArray[:,:,1]
    b = imArray[:,:,2]


    print r.shape

    r = np.uint8(np.real(fftpack.fft2(r)))
    g = np.uint8(np.real(fftpack.fft2(g)))
    b = np.uint8(np.real(fftpack.fft2(b)))

    r = cv.GaussianBlur(r,(3,3),0)
    g = cv.GaussianBlur(g,(3,3),0)
    b = cv.GaussianBlur(b,(3,3),0)

    ret, r = cv.threshold(r,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, g = cv.threshold(g,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, b = cv.threshold(b,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    r = cv.GaussianBlur(r,(11,11),0)
    g = cv.GaussianBlur(g,(11,11),0)
    b = cv.GaussianBlur(b,(11,11),0)
    
    ret, r = cv.threshold(r,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, g = cv.threshold(g,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, b = cv.threshold(b,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    r = cv.GaussianBlur(r,(21,21),0)
    g = cv.GaussianBlur(g,(21,21),0)
    b = cv.GaussianBlur(b,(21,21),0)
    
    ret, r = cv.threshold(r,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, g = cv.threshold(g,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, b = cv.threshold(b,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    r = cv.GaussianBlur(r,(31,31),0)
    g = cv.GaussianBlur(g,(31,31),0)
    b = cv.GaussianBlur(b,(31,31),0)
    
    ret, r = cv.threshold(r,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, g = cv.threshold(g,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, b = cv.threshold(b,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


    imArray2 = np.ones((imArray.shape[0],imArray.shape[1],3)) * 200

    imArray2[:,:,0] = r
    imArray2[:,:,1] = g
    imArray2[:,:,2] = b

    print imArray2.shape
    print imArray2

    print np.max(np.real(r))
    print np.min(np.real(r))

    print np.min (np.uint8(imArray2))
    print np.max (np.uint8(imArray2))

    im2 = Image.fromarray(np.uint8(imArray2), mode = 'RGB')

    im2.save('output/transforms/%s.png' % i)

    imArray = np.asarray(im2)

    print np.min(imArray)
    print np.max(imArray)

"""
for i in range:

    print i

    imArray = np.asarray(im)


    r = imArray[:,:,0]
    g = imArray[:,:,1]
    b = imArray[:,:,2]

    resultr = (np.fft.fft2(r))
    resultg = (np.fft.fft2(g))
    resultb = (np.fft.fft2(b))

    resultrI = np.imag(resultr) 
    resultrR = np.real(resultr) 
    
    resultrI = ndimage.filters.gaussian_filter(resultrR, i, mode = 'wrap')
    #resultrR = ndimage.filters.gaussian_filter(resultrR, 100, mode = 'wrap')
    
    resultr  =  resultrR + (resultrI * 1j ) # * i/30.0)
    
    resultgI = np.imag(resultg) 
    resultgR = np.real(resultg) 

    resultgI = ndimage.filters.gaussian_filter(resultgI, i, mode = 'wrap')
    
    resultg  =  resultgR + (resultgI * 1j  ) # * i/30.0)
    
    resultbI = np.imag(resultb) 
    resultbR = np.real(resultb) 

    resultbI = ndimage.filters.gaussian_filter(resultbI, i, mode = 'wrap')
    
    resultb  =  resultbR + (resultbI * 1j ) # * i/30.0)

    r2 = np.rint(np.real((np.fft.ifft2(resultr))))
    g2 = np.rint(np.real((np.fft.ifft2(resultg))))
    b2 = np.rint(np.real((np.fft.ifft2(resultb))))

    imArray2 = np.zeros((imArray.shape[0],imArray.shape[1],3))

    imArray2[:,:,0] = r2
    imArray2[:,:,1] = g2
    imArray2[:,:,2] = b2

    im2 = Image.fromarray(np.uint8(imArray2), mode = 'RGB')


    im3 = ImageOps.invert(im2)

    im2.save('output/blockyFFTout%s.png' % i)
    #im3.save('output/blockyInv%s.png' % i)
    """