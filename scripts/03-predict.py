import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
#from constants import *
from model_bce import ModelBCE
import sys


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('\\')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print os.path.join(path_to_images, curr_file + '.jpg')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)


def main(in_folder, out_folder):
    # Create network
    model = ModelBCE(256, 192, batch_size=8)
    # Here need to specify the epoch of model sanpshot
    load_weights(model.net['output'], path= 'ft45_18_model6_gen_', epochtoload=150, layernum=54)
    # Here need to specify the path to images and output path
    test(path_to_images=in_folder, path_output_maps=out_folder, model_to_test=model)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise NotImplementedError
    elif len(sys.argv) == 3:
        image_fol = sys.argv[1]
        print 'Image folder is %s' % image_fol
        output_fol = sys.argv[2]
        print 'Saliency folder is %s' % output_fol
        main(image_fol, output_fol)
    else:
        raise NotImplementedError
