import numpy as np
from PIL import Image
import os

def edgeMatch(image1, image2, direction):

    #0 is right, 1 is down, 2 is left, 3 is up
    
    
    
    if(direction == 0):
        
        
        
        slice1 = image1[ 0 : image1.shape[0], image1.shape[1] - 100 : image1.shape[1], :]
        
        slice2 = image2[  0 : image1.shape[0], 0 : 100, :]
        
        combine = np.zeros((image1.shape[0], 200, 3))
        
        combine[:,0:100,:] = slice1
        combine[:,100:200,:] = slice2
        
        
        slice1 = Image.fromarray(slice1)
        slice2 = Image.fromarray(slice2)
        combine = Image.fromarray(np.uint8(combine))
        
        slice1.save('slice1.png')
        slice2.save('slice2.png')
        combine.save('combine.png')
        
image1 = np.asarray(Image.open('output/transforms/380.png'))
image2 = np.asarray(Image.open('output/transforms/381.png'))


def compareTiles(tile1, tile2):

    diff = (tile1 - tile2) ** 2
    
    return np.sum(diff)
    
def getTiles(rootdir):

    tiles = []

    for subdir, dirs, files in os.walk(rootdir):
    
        #print files
   
        for file in files:
            
            if(file[-3:] == 'png'):
                print file
                tiles.append(np.asarray(Image.open(rootdir + "/" + file)))
            
    return tiles
    

def buildmatch(tiles, target, composite):

    bestScore = 9223372036854775807
    tilenum = 0
    count = 0

    for tile in tiles:
        nextComp = np.uint8(composite + tile)
        score = compareTiles(nextComp, target)
        if (score < bestScore):
            bestScore = score
            tilenum = count
        count = count + 1
        
    for tile in tiles:
        nextComp = np.uint8(composite - tile)
        score = compareTiles(nextComp, target)
        if (score < bestScore):
            bestScore = score
            tilenum = count
        count = count + 1
        
    if(tilenum < len(tiles)):
        return np.uint8(composite + tiles[tilenum])
    elif(tilenum >= len(tiles)):
        return np.uint8(composite + tiles[tilenum - len(tiles)])
    else:
        die = 1/0


target = np.asarray(Image.open('circle.png')) # needs to be 512x512

print target.shape

composite = np.zeros_like(target)
        
tiles = getTiles('output/transforms')

print len(tiles)
        
for i in range(1000):
    
    print i
    
    composite = buildmatch(tiles, target, composite)
    
    Image.fromarray(composite).save('compositeOut/%s.png' %i)
