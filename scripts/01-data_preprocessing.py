# Process raw data and save them into pickle file.
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
from constants import *
import pdb

img_size = INPUT_SIZE
salmap_size = INPUT_SIZE

# Resize train/validation files
listImgFiles = [k.split('\\')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*'))]
listTestImages = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]


'''
for currFile in tqdm(listImgFiles):
    tt = dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'),
                                   os.path.join(pathToMaps, currFile + '.png'),
                                   os.path.join(pathToFixationMaps, currFile + '.mat'),
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)

    # if tt.image.getImage().shape[:2] != (480, 640):
    #    print 'Error:', currFile

    imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), img_size, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2BGR)
    saliencyResized = cv2.resize(tt.saliency.getImage(), salmap_size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)
    cv2.imwrite(os.path.join(pathOutputMaps, currFile + '.png'), saliencyResized)

# Resize test files

for currFile in tqdm(listTestImages):
    tt = dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'),
                                   os.path.join(pathToMaps, currFile + '.mat'),
                                   os.path.join(pathToFixationMaps, currFile + '.mat'),
                                   dataRepresentation.LoadState.loaded,dataRepresentation.InputType.image,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)

    imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), img_size, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)

'''
# LOAD DATA

# Train

#listFilesTrain = [k for k in listImgFiles if 'train' in k and np.mod(int(k[10]),3)==1 and np.mod(int(k[11]),3)==1]
listFilesTrain = [k for k in listImgFiles if 'train_' in k]

trainData = []
for currFile in tqdm(listFilesTrain):
    trainData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.jpg'),
                                               os.path.join(pathOutputMaps, currFile + '.png'),
                                               os.path.join(pathToFixationMaps, currFile + '.png'),
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMap))    

with open(os.path.join(pathToPickle, 'train_with_fixa.pickle'), 'wb') as f:      # 45 degree, every 10 frames, data of 2018
    pickle.dump(trainData, f)

# Validation
listFilesValidation = [k for k in listImgFiles if 'val_' in k]

validationData = []
for currFile in tqdm(listFilesValidation):
    validationData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.jpg'),
                                                    os.path.join(pathOutputMaps, currFile + '.png'),
                                                    os.path.join(pathToFixationMaps, currFile + '.png'),
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMap))
                                                    
with open(os.path.join(pathToPickle, 'validation_with_fixa.pickle'), 'wb') as f:
    pickle.dump(validationData, f)

# Test

testData = []

for currFile in tqdm(listTestImages):
    testData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.jpg'),
                                              os.path.join(pathOutputMaps, currFile + '.png'),
                                              dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                              dataRepresentation.LoadState.unloaded,
                                              dataRepresentation.InputType.empty))

with open(os.path.join(pathToPickle, 'testData.pickle'), 'wb') as f:
    pickle.dump(testData, f)
