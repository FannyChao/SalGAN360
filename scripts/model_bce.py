import lasagne
from lasagne.layers import InputLayer
import theano
import theano.tensor as T
import numpy as np

import generator
from model import Model


def KL_div(output, target):
    #output = output/output.max()
    #target = target/target.max()
    output = output/output.sum()
    target = target/target.sum()    
    a = target * T.log(target/(output+1e-50)+1e-50)
    b = output * T.log(output/(target+1e-50)+1e-50)
    return (a.sum()+b.sum())/(2.)

def CC(output, target):
    #output = output/output.max()
    #target = target/target.max()
    #output = output/output.sum()
    #target = target/target.sum()
    
    output = (output-output.mean())/output.std()
    target = (target-target.mean())/target.std()
    
    num=(output-output.mean())*(target-target.mean())
    out_squre = T.square(output-output.mean())
    tar_squre = T.square(target-target.mean())
    
    return num.sum()/(T.sqrt(out_squre.sum()*tar_squre.sum()))
    
def NSS(output, fixationMap):
    #pdb.set_trace()
    #output = cv2.resize(output, (fixationMap.shape), interpolation=cv2.INTER_CUBIC )
    output = (output-output.mean())/output.std()
    Sal = output*fixationMap
    return Sal.sum()/fixationMap.sum()   



class ModelBCE(Model):
    def __init__(self, w, h, batch_size=32, lr=0.0001): #fy: change from 0.001
        super(ModelBCE, self).__init__(w, h, batch_size)

        self.net = generator.build(self.inputHeight, self.inputWidth, self.input_var)

        output_layer_name = 'output'
        prediction = lasagne.layers.get_output(self.net[output_layer_name])

        test_prediction = lasagne.layers.get_output(self.net[output_layer_name], deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        output_var_sal_pooled = T.signal.pool.pool_2d(self.output_var_sal, (4, 4), mode="average_exc_pad", ignore_border=True)
        output_var_fixa_pooled = T.signal.pool.pool_2d(self.output_var_fixa, (4, 4), mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)

        #bce = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_pooled).mean()
        #train_err = bce
        KLmiu = 2.4948
        KLstd = 1.7421
        CCmiu = 0.3932
        CCstd = 0.2565
        NSSmiu = 0.4539
        NSSstd = 0.2631
        bcemiu = 0.3194
        bcestd = 0.1209
        #train_err = bcemiu+bcestd*((3.)*((KL_div(prediction_pooled, output_var_sal_pooled)-KLmiu)/KLstd) - (1.)*((CC(prediction_pooled, output_var_sal_pooled)-CCmiu)/CCstd) - (1.)*((NSS(prediction_pooled, output_var_fixa_pooled)-NSSmiu)/NSSstd))
        train_err = (1.)*(KL_div(prediction_pooled, output_var_sal_pooled)) - (1.)*((CC(prediction_pooled, output_var_sal_pooled))) - (1.)*((NSS(prediction_pooled, output_var_fixa_pooled)))

        # parameters update and training
        G_params = lasagne.layers.get_all_params(self.net[output_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(lr, dtype=theano.config.floatX))
        G_updates = lasagne.updates.nesterov_momentum(train_err, G_params, learning_rate=self.G_lr, momentum=0.5)

        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var_sal, self.output_var_fixa], outputs=train_err, updates=G_updates,
                                               allow_input_downcast=True)
