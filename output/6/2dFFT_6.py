from PIL import Image, ImageOps, ImageDraw
import numpy as np
from scipy import ndimage, fftpack
import cv2 as cv
import os
import sys
from shutil import copyfile

#https://stackoverflow.com/questions/2598734/numpy-creating-a-complex-array-from-2-real-ones

#def subpixel(
   
directory = "output/%s/" % sys.argv[1]
   
if not os.path.exists(directory):
    os.makedirs(directory)
    
copyfile(sys.argv[0], directory + sys.argv[0])
    
directory2 = "output/%s/input" % sys.argv[1]
   
if not os.path.exists(directory2):
    os.makedirs(directory2)
    
directory3 = "output/%s/transform" % sys.argv[1]
   
if not os.path.exists(directory3):
    os.makedirs(directory3)
    
    
def drawCenterSquarePoint(draw, number, color):

    

    x = number / 4 + 2  # + 1 to map 5x5 co-ordinate to 7x7 framework, and also + 1 to apply buffer unit for circle overflow
    y = number % 4 + 2
    
    #x = x + array.shape[0]/2 - 1
    #y = y + array.shape[0]/2 - 1
    
    #  magic number ref:  791 - 42 - 42 =  707
    # grid is 7 x 7, each cell is 101 pixels
    # 42 on each side as buffer
    
    x = x * 101
    y = y * 101
    
    
    if(color == 1):
        draw.ellipse([(x-143, y-143),(x+143,y+143)],fill=(255, 255, 200))
    if(color == 2):
        draw.ellipse([(x-143, y-143),(x+143,y+143)],fill=(255, 200, 255))
    if(color == 3):
        draw.ellipse([(x-143, y-143), (x+143,y+143)],fill=(200, 255, 255))
        
def drawCenterSquarePoint2(draw, number, color, epsilon):

    

    x = number / 4 + 2  # + 1 to map 5x5 co-ordinate to 7x7 framework, and also + 1 to apply buffer unit for circle overflow
    y = number % 4 + 2
    
    #x = x + array.shape[0]/2 - 1
    #y = y + array.shape[0]/2 - 1
    
    #  magic number ref:  791 - 42 - 42 =  707
    # grid is 7 x 7, each cell is 101 pixels
    # 42 on each side as buffer
    
    x = x * 101 + epsilon
    y = y * 101
    
    
    if(color == 1):
        draw.ellipse([(x-143, y-143),(x+143,y+143)],fill=(255, 255, 250))
    if(color == 2):
        draw.ellipse([(x-143, y-143),(x+143,y+143)],fill=(255, 250, 255))
    if(color == 3):
        draw.ellipse([(x-143, y-143), (x+143,y+143)],fill=(250, 255, 255))
    

def drawPoint(array, number, color, num):

    x = number / 4
    y = number % 4
    
    x = x + array.shape[0]/2 - 1
    y = y + array.shape[0]/2 - 1
    
    if(color == 1):
        array[x,y, :] = [255, 255, 200 - num / 5.0]
    if(color == 2):
        array[x,y, :] = [200, 255, 255 - num / 5.0]
    if(color == 3):
        array[x,y, :] = [255, 200, 255 - num / 5.0]
        

def hartley( n ):

  import numpy as np

  a = np.zeros ( ( n, n ) )

  for i in range ( 0, n ):
    for j in range ( 0, n ):

      angle = 2.0 * np.pi * float ( i * j ) / float ( n )

      a[i,j] = np.sin ( angle ) + np.cos ( angle )

  return a

for i in range(16 ** 3):

    centerSquare = Image.new('RGB',(909,909),'black')
    draw = ImageDraw.Draw(centerSquare)

  
    #im = Image.open('blocky_2.png')

    #range = np.linspace(.28,.04,50)

    #imArray = np.asarray(im)

    #imArray = np.zeros((512,512,3))

    #editIm = Image.fromarray(np.uint8(imArray))

    #draw = ImageDraw.Draw(editIm)
    
    base = 16
    
    #drawCenterSquarePoint(draw, i % base, 1)
    #drawCenterSquarePoint(draw, (i / base) % base, 2)
    #drawCenterSquarePoint2(draw, ((i / base) / base) % base, 3, i)
    
    drawCenterSquarePoint(draw, i % base, 1)
    drawCenterSquarePoint(draw, (i / base) % base, 2)
    drawCenterSquarePoint(draw, ((i / base) / base) % base, 3)
    
    
    centerSquare.save(directory2 + '/%s.png' %i)
    
    centerSquare.thumbnail((9,9), Image.LANCZOS)
    
    
    
    
    midArray = np.asarray(centerSquare)
    
    print midArray
    
    imArray = np.zeros((512,512,3))
    
    imArray[512/2-5:512/2+4,512/2-5:512/2+4] = midArray
    
    
    
    
    
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

    #im.save('output/inputs_color6/%s.png' %i)

    r = imArray[:,:,0]
    g = imArray[:,:,1]
    b = imArray[:,:,2]


    #print r.shape

    r = np.uint8(np.real(fftpack.fft2(r)))
    g = np.uint8(np.real(fftpack.fft2(g)))
    b = np.uint8(np.real(fftpack.fft2(b)))

    
    r = (cv.GaussianBlur(r,(17,17),0) + cv.GaussianBlur(r,(61,61),0)) #+ cv.GaussianBlur(r,(71,71),0 ))/ 3
    g = (cv.GaussianBlur(g,(17,17),0) + cv.GaussianBlur(g,(61,61),0)) # + cv.GaussianBlur(g,(71,71),0 ))/ 3
    b = (cv.GaussianBlur(b,(17,17),0) + cv.GaussianBlur(b,(61,61),0)) # + cv.GaussianBlur(b,(71,71),0 ))/ 3
    
    
    ret, r2 = cv.threshold(r,100,255,cv.THRESH_BINARY_INV)#+cv.THRESH_OTSU)
    ret, g2 = cv.threshold(g,100,255,cv.THRESH_BINARY_INV)#+cv.THRESH_OTSU)
    ret, b2 = cv.threshold(b,100,255,cv.THRESH_BINARY_INV)#+cv.THRESH_OTSU)
    
    r = r2#(r + r2) / 2
    g = g2#(g + g2) / 2
    b = b2#(b + b2) / 2 
    
    #r = cv.GaussianBlur(r,(11,11),0)
    #g = cv.GaussianBlur(g,(11,11),0)
    #b = cv.GaussianBlur(b,(11,11),0)
    
    #ret, r = cv.threshold(r,255,100,cv.THRESH_TRUNC)
    #ret, g = cv.threshold(g,255,100,cv.THRESH_TRUNC)
    #ret, b = cv.threshold(b,255,100,cv.THRESH_TRUNC)
    
    """
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
    """
    #r = cv.GaussianBlur(r,(11,11),0)
    #g = cv.GaussianBlur(g,(11,11),0)
    #b = cv.GaussianBlur(b,(11,11),0)

    
    #c = np.rint(r * 151.11 + g * 202.22 + b * 253.33) 
    
    #r2 = c % 256 #c 
    #g2 = (c / 4 ) % 256#(c/2 % 64) * 4 
    #b2 = (c / 8 ) % 256 #(c/4 % 64) * 4


    imArray2 = np.ones((imArray.shape[0],imArray.shape[1],3)) * 200

    imArray2[:,:,0] = r
    imArray2[:,:,1] = g
    imArray2[:,:,2] = b

    #print imArray2.shape
    #print imArray2

    #print np.max(np.real(r))
    #print np.min(np.real(r))

    #print np.min (np.uint8(imArray2))
    #print np.max (np.uint8(imArray2))

    im2 = Image.fromarray(np.uint8(imArray2), mode = 'RGB')

    im2.save(directory3 +'/%s.png' % i)

    imArray = np.asarray(im2)

    #print np.min(imArray)
    #print np.max(imArray)

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