# SalGAN360: Visual Saliency Detection on 360° images with GAN

- This repo contains the codes that used in paper [*SalGAN360: Visual Saliency Detection on 360° images with GAN (ICMEW 2018)*](http://openhevc.insa-rennes.fr/wp-content/uploads/2018/07/camera-ready_icme2018template.pdf) by Fang-Yi Chao, Lu Zhang, Wassim Hamidouche, Olivier Deforges.
- The winner in Prediction of Head+Eye Saliency for Images in [*Salient360! Grand Challenges at ICME’18*.](https://salient360.ls2n.fr/) 

## Abstract
Understanding visual attention of observers on 360° images gains interest along with the booming trend of Virtual Reality applications. Extending existing saliency prediction methods from traditional 2D images to 360° images is not a direct approach due to the lack of a sufficient large 360° image saliency  database. In  this  paper,  we  propose  to  extend  the SalGAN, a 2D saliency model based on the generative adversarial network, to SalGAN360 by fine tuning the SalGAN with our new loss function to predict both global and local saliency maps.  Our experiments show that the SalGAN360 outperforms the tested state-of-the-art methods.

## Visual Results
![qualitative results](https://github.com/FannyChao/SalGAN360/blob/master/figs/result_image360.jpg)


## Requirements
- Download [SalGAN](https://github.com/imatge-upc/saliency-salgan-2017)
- Matlab

## Pretrained models
- [SalGAN360 Generator Model](https://drive.google.com/open?id=1YRZQJTynqfaZmLYgbJPZFYLFf4_jSlv_)


## Usage
Replace ```01-data_preprocessing.py```, ```02-train.py```, ```03-predict.py```, ```model_salgan.py```, ```dataRepresentation.py```, ```model.py``` and ``` utils.py ``` in SalGAN. 
- Test: To predict saliency maps, run ```salgan360.m``` after specifying the path to images and the path to the output saliency maps
- Train: 
   - 1. Run ```preprocessing_trainingdata.m``` to transfer 360° images into multiple viewports.
   - 2. Run ```01-data_preprocessing.py``` to make pickle files.
   - 3. Run ```02-train.py``` to fine tune salgan model.

## Citing
```
   @INPROCEEDINGS{8551543,
       author = {F. Chao and L. Zhang and W. Hamidouche and O. Deforges},
       booktitle = {2018 IEEE International Conference on Multimedia Expo Workshops (ICMEW)},
       title = {Salgan360: Visual Saliency Prediction On 360 Degree Images With Generative Adversarial Networks},
       year = {2018},
       month = {July},}
```
