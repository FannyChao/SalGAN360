import lasagne
from lasagne.layers import InputLayer
import theano
import theano.tensor as T
import numpy as np

import generator
import discriminator
from model import Model
import pdb
import cv2
from scipy.stats.stats import pearsonr


def KL_div(output, target):
    output = output/output.sum()
    target = target/target.sum()    
    a = target * T.log(target/(output+1e-20)+1e-20)
    b = output * T.log(output/(target+1e-20)+1e-20)
    return (a.sum()+b.sum())/(2.)

def CC(output, target):
    output = (output-output.mean())/output.std()
    target = (target-target.mean())/target.std()
    num=(output-output.mean())*(target-target.mean())
    out_square = T.square(output-output.mean())
    tar_square = T.square(target-target.mean())
    CC_score = num.sum()/(T.sqrt(out_square.sum()*tar_square.sum()))
    if T.isnan(CC_score):
        CC_score = 0        
    return CC_score
    
def NSS(output, fixationMap):
    output = (output-output.mean())/output.std()
    Sal = output*fixationMap
    NSS_score = Sal.sum()/fixationMap.sum()
    if T.isnan(NSS_score):
        NSS_score = 0
    return NSS_score   


class ModelSALGAN(Model):
    def __init__(self, w, h, batch_size=16, G_lr=3e-5, D_lr=3e-5, alpha=1/20.):
        super(ModelSALGAN, self).__init__(w, h, batch_size)

        # Build Generator
        self.net = generator.build(self.inputHeight, self.inputWidth, self.input_var)

        # Build Discriminator
        self.discriminator = discriminator.build(self.inputHeight, self.inputWidth,
                                                 T.concatenate([self.output_var_sal, self.input_var], axis=1))

        # Set prediction function
        output_layer_name = 'output'

        prediction = lasagne.layers.get_output(self.net[output_layer_name])
        test_prediction = lasagne.layers.get_output(self.net[output_layer_name], deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        disc_lab = lasagne.layers.get_output(self.discriminator['prob'],
                                             T.concatenate([self.output_var_sal, self.input_var], axis=1))
        disc_gen = lasagne.layers.get_output(self.discriminator['prob'],
                                             T.concatenate([prediction, self.input_var], axis=1))

        # Downscale the saliency maps
        output_var_sal_pooled = T.signal.pool.pool_2d(self.output_var_sal, (4, 4), mode="average_exc_pad", ignore_border=True)
        output_var_fixa_pooled = T.signal.pool.pool_2d(self.output_var_fixa, (4, 4), mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)
        '''
        ICME17 image dataset
        KLmiu = 2.4948
        KLstd = 1.7421
        CCmiu = 0.3932
        CCstd = 0.2565
        NSSmiu = 0.4539
        NSSstd = 0.2631
        bcemiu = 0.3194
        bcestd = 0.1209
        '''
        #ICME18 image dataset
        KLmiu = 2.9782 
        KLstd = 2.1767
        CCmiu = 0.3677 
        CCstd = 0.2484
        NSSmiu = 0.5635
        NSSstd = 0.2961
        bcemiu = 0.2374
        bcestd = 0.1066
          
        #model1
        #train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_sal_pooled).mean()
        #model6 
        train_err = bcemiu+bcestd*((1.)*((KL_div(prediction_pooled, output_var_sal_pooled)-KLmiu)/KLstd) - (1.)*((CC(prediction_pooled, output_var_sal_pooled)-CCmiu)/CCstd) - (1.)*((NSS(prediction_pooled, output_var_fixa_pooled)-NSSmiu)/NSSstd))
        #model8
        #train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_sal_pooled).mean()-(bcemiu+bcestd*((1.)*((CC(prediction_pooled, output_var_sal_pooled)-CCmiu)/CCstd) + (1.)*((NSS(prediction_pooled, output_var_fixa_pooled)-NSSmiu)/NSSstd)))
        + 1e-4 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)
        #pdb.set_trace()
        # Define loss function and input data
        ones = T.ones(disc_lab.shape)
        zeros = T.zeros(disc_lab.shape)
        D_obj = lasagne.objectives.binary_crossentropy(T.concatenate([disc_lab, disc_gen], axis=0),
                                                       T.concatenate([ones, zeros], axis=0)).mean()
        #D_obj = bcemiu+bcestd*((3.)*((KL_div(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0)).sum()-KLmiu)/KLstd) - (1.)*((CC(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0))-CCmiu)/CCstd) - (1.)*((NSS(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0))-NSSmiu)/NSSstd))
        #D_obj = (3.)*((KL_div(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0)).sum()-KLmiu)/KLstd)
        + 1e-4 * lasagne.regularization.regularize_network_params(self.discriminator['prob'], lasagne.regularization.l2)

        G_obj_d = lasagne.objectives.binary_crossentropy(disc_gen, T.ones(disc_lab.shape)).mean()
        #G_obj_d = bcemiu+bcestd*((3.)*((KL_div(disc_gen, T.ones(disc_lab.shape)).sum()-KLmiu)/KLstd) - (1.)*((CC(disc_gen, T.ones(disc_lab.shape))-CCmiu)/CCstd) - (1.)*((NSS(disc_gen, T.ones(disc_lab.shape))-NSSmiu)/NSSstd))
        + 1e-4 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)

        G_obj = G_obj_d + train_err * alpha
        cost = [G_obj, D_obj, train_err]

        # parameters update and training of Generator
        G_params = lasagne.layers.get_all_params(self.net[output_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(G_lr, dtype=theano.config.floatX))
        G_updates = lasagne.updates.adagrad(G_obj, G_params, learning_rate=self.G_lr)
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var_sal, self.output_var_fixa], outputs=cost,                                     
                                               updates=G_updates, allow_input_downcast=True,  on_unused_input='ignore')

        # parameters update and training of Discriminator
        D_params = lasagne.layers.get_all_params(self.discriminator['prob'], trainable=True)
        self.D_lr = theano.shared(np.array(D_lr, dtype=theano.config.floatX))
        D_updates = lasagne.updates.adagrad(D_obj, D_params, learning_rate=self.D_lr)
        self.D_trainFunction = theano.function([self.input_var, self.output_var_sal, self.output_var_fixa], cost, updates=D_updates,                                       
                                               allow_input_downcast=True,  on_unused_input='ignore')

