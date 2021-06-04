from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
try:
    #Support for tensorflow <= 2.3
    from tensorflow.python.keras.engine.training import _minimize
except ImportError:
    pass
import json
import os
import pickle
import h5py

SMALL_CONSTANT = 1e-12
GAUSS_RANGE = 10.0

def safe_abs(val):
    ret = tf.math.maximum(tf.abs(val), K.epsilon())
    return ret

class GaussianMixture(keras.layers.Layer):
    def __init__(self, n_draws=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_draws = n_draws
    
    def get_distribution(self, inputs):
        #Abbreviations: bs = batch size, nm = number of mixture components, ld = latent space dimension
        #Expected input shape [(bs, nm, ld), (bs, nm, ld), (bs, nm)]
        mean = inputs[0]
        var = SMALL_CONSTANT + tf.exp(inputs[1])
        return tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=inputs[2]),
                                     components_distribution=tfd.MultivariateNormalDiag(loc=mean,
                                                                                        scale_diag=tf.sqrt(var)))
    
    def call(self, inputs):
        #Expected input shape [(batch size, number of mixture components, latent space dimension, 2), (batch size, number of mixture components)]
        dist = self.get_distribution(inputs)
        return dist.sample(self.n_draws)
    
    def set_ramp(self, ramp):
        return

class MultiDistribution(keras.layers.Layer):
    def __init__(self, distributions, n_draws=1, base_args=None,
                 dist_dict=None, ramp=1., *args, **kwargs):
        """
        Arguments
        ---------
        distributions : list of tuple
            A list of tuples of size 2. The first entry has to be a
            string that specifies a distribution. (See get distribution
            for more info) The second entry has to be an iterable of
            strings or an iterable of tuples. See [1] in the Notes below
            for more info.
        n_draws : {int, 0}
            How many samples should be drawn on each pass.
        base_args : {None or list of None or dict, None}
            Fixed arguments that should be passed to the distributions.
            The index in this list must line up with the distributions
            given in `distributions`.
        dist_dict : {None or dict, None}
            A dictionary containing key-value-pairs where each key is
            the name to a distribution and the value is a tensorflow
            probability distribution. These names are then accessible by
            the `distributions` argument. If set to None, no additional
            distributions will be used.
        ramp : {float, 1.}
            A float between 0 and 1. Defines the training ramp. This
            value is automatically added to the base_args if possible.
        
        Notes
        -----
        [1]:
            If an item from the iterable is of type string, it will be
            converted to `(str, (1, ))`. If an item is of type tuple, it
            has to contain two entries. The first has to be a string
            specifying a keyword-argument for the corresponding
            distribution. The second has to specify a shape.
        """
        super().__init__(*args, **kwargs)
        self.distributions = distributions
        if base_args is None:
            self.base_args = [{} for _ in self.distributions]
        else:
            self.base_args = [{} if pt is None else pt for pt in base_args]
        self.n_draws = n_draws
        self.dist_dict = {'gauss': tfd.Normal,
                          'gauss_valid': lambda loc, scale: tfd.Normal(loc=loc, scale=safe_abs(scale), validate_args=True),
                          'truncGauss': tfd.TruncatedNormal,
                          'vonMises': tfd.VonMises,
                          'vonMisesFisher': tfd.VonMisesFisher}
        if dist_dict is not None:
            self.dist_dict.update(dist_dict)
        self.ramp = tf.Variable(1.0, dtype=tf.float32, name='ramp_'+self.name,
                                trainable=False)
        self.set_ramp(ramp)
    
    def build(self, input_shape):
        return
    
    def get_distribution_by_name(self, name):
        return self.dist_dict[name]
    
    def get_distribution(self, inputs):
        idx = 0
        dists = []
        for i, (dist_name, arg_list) in enumerate(self.distributions):
            dist = self.get_distribution_by_name(dist_name)
            kwargs = {}
            for kwarg in arg_list:
                if isinstance(kwarg, str):
                    kwarg = (kwarg, (1, ))
                key, shape = kwarg
                if shape is None:
                    length = 1
                    shape = [tf.shape(inputs)[0]]
                else:
                    length = np.prod(shape)
                    shape = [tf.shape(inputs)[0]] + list(shape)
                arg = inputs[:,idx:idx+length]
                arg = tf.reshape(arg, shape)
                idx += length
                kwargs[key] = arg
            kwargs.update(self.base_args[i])
            ramp_kwargs = kwargs.copy()
            ramp_kwargs.update({'ramp': self.ramp})
            try:
                curr = dist(**ramp_kwargs)
            except TypeError:
                print("Unable to use ramp for {}".format(dist_name))
                curr = dist(**kwargs)
            dists.append(curr)
        return tfd.JointDistributionSequential(dists)
    
    def call(self, inputs):
        #Input must be a 1D-tensor (or 2D, if first axis is batch)
        joint = self.get_distribution(inputs)
        return joint.sample(self.n_draws)
    
    def set_ramp(self, ramp):
        self.ramp.assign(ramp)

class MultiVariateNormal(keras.layers.Layer):
    def __init__(self, n_draws=1, *args, **kwargs):
        """
        Arguments
        ---------
        n_draws : {int, 0}
            How many samples should be drawn on each pass.
        """
        super().__init__(*args, **kwargs)
        self.n_draws = n_draws
    
    def get_distribution(self, inputs):
        #Abbreviations: bs = batch size, nm = number of mixture components, ld = latent space dimension
        #Expected input shape [(bs, ld), (bs, ld)]
        mean = inputs[0]
        var = SMALL_CONSTANT + tf.exp(inputs[1])
        return tfd.MultivariateNormalDiag(loc=mean,
                                          scale_diag=tf.sqrt(var))
    
    def call(self, inputs):
        #Input must be two 1D-tensors (or 2D, if first axis is batch)
        joint = self.get_distribution(inputs)
        return joint.sample(self.n_draws)
    
    def set_ramp(self, ramp):
        return
        

class CVAE(keras.models.Model):
    def __init__(self, E1, E2, D, output_distributions, n_latent_draws=1,
                 dist_dict=None, ramp=1., trainable_ramp=False):
        super().__init__()
        #Expected input shape  E1: [(batch size, 2048, num detectors)]
        #Expected output shape E1: [(batch size, num mixture components, latent space size), (batch size, num mixture components, latent space size), (batch size, num mixture components)]
        self.E1 = E1
        #Expected input shape  E2: [(batch size, num params), (batch size, 2048, num detectors)]
        #Expected output shape E2: [(batch size, latent space size), (batch size, latent space size)]
        self.E2 = E2
        #Expected input shape   D: [(batch size, 2048, num detectors), (batch size, latent space size)]
        #Expected output shape  D: (2 * num of inferred components)
        self.D  = D
        self.n_latent_draws = n_latent_draws
        self.output_distributions = output_distributions
        self.E1mixture = GaussianMixture(n_draws=self.n_latent_draws)
        self.E2dist = MultiVariateNormal(n_draws=self.n_latent_draws)
        self.Dout = MultiDistribution(output_distributions,
                                      dist_dict=dist_dict)
        self.ramp = tf.Variable(1.0, dtype=tf.float32,
                                trainable=trainable_ramp)
        self.set_ramp(ramp)
    
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            #Forward pass
            distE1params = self.E1(x[1])
            distE1 = self.E1mixture.get_distribution(distE1params)
            
            distE2params = self.E2(x)
            distE2 = self.E2dist.get_distribution(distE2params)
            
            z = distE2.sample(self.n_latent_draws)
            
            eps = tf.random.normal(tf.shape(x[1]))
            y_ramp = x[1] * self.ramp + eps * (1 - self.ramp)
            distDparams = self.D([y_ramp, tf.squeeze(tf.concat(z, axis=-1))])
            distD = self.Dout.get_distribution(distDparams)
            y_samp = distD.sample()
            y_pred = tf.concat(y_samp, axis=-1)
            
            #Calculate loss
            kl1 = tf.squeeze(distE1.log_prob(tf.squeeze(tf.concat(z, axis=-1))))
            kl2 = tf.squeeze(distE2.log_prob(z))
            kl = kl2 - kl1
            
            prep_labels = [x[0][:,i] for i in range(x[0].shape[-1])]
            prep_labels = [tf.expand_dims(pt, axis=-1) for pt in prep_labels]
            prep_labels = [tf.expand_dims(pt, axis=0) for pt in prep_labels]
            labels = []
            curr_idx = 0
            shapes = []
            for batch, event in zip(distD.batch_shape, distD.event_shape):
                if batch == tf.TensorShape(None):
                    batch = [None]
                else:
                    batch = batch.as_list()
                if event == tf.TensorShape(None):
                    event = [None]
                else:
                    event = event.as_list()
                shapes.append(batch + event)
            for i, pt in enumerate(y_samp):
                if shapes[i][-1] == 1:
                    labels.append(prep_labels[curr_idx])
                    curr_idx += 1
                else:
                    labels.append(tf.concat(prep_labels[curr_idx:curr_idx+shapes[i][-1]], axis=-1))
                    curr_idx += shapes[i][-1]
            
            elbo = -tf.squeeze(distD.log_prob(labels))
            loss = tf.reduce_mean(elbo + self.ramp * kl, axis=0)
        
        for_metrics = {'-ELBO': tf.reduce_mean(elbo, axis=0),
                       'KL-Divergence': tf.reduce_mean(kl, axis=0),
                       'ramp': self.ramp,
                       'loss': loss}
        try:
            #Support for tensorflow <= 2.3
            _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.trainable_variables)
        except NameError:
            self.optimizer.minimize(loss, self.trainable_variables,
                                    tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        tmp = {m.name: m.result() for m in self.metrics}
        tmp.update(for_metrics)
        return tmp
    
    #@tf.function
    def call(self, inputs, training=False):
        if training:
            #Expecting inputs: [x, Y]
            distE2params = self.E2(inputs)
            distE2 = self.E2dist.get_distribution(distE2params)
            z = distE2.sample(self.n_latent_draws)
            distDparams = self.D([inputs[1], tf.squeeze(tf.concat(z, axis=-1))])
            distD = self.Dout.get_distribution(distDparams)
            y_samp = distD.sample()
            y_pred = tf.concat(y_samp, axis=-1)
            return y_pred
        else:
            #Expecting input: [x, Y]
            distE1params = self.E1(inputs[1])
            distE1 = self.E1mixture.get_distribution(distE1params)
            z = distE1.sample(self.n_latent_draws)
            distDparams = self.D([inputs[1], tf.squeeze(tf.concat(z, axis=-1))])
            return tf.concat(self.Dout(distDparams), axis=-1)
            distD = self.Dout.get_distribution(distDparams)
            y_samp = distD.sample()
            y_pred = tf.concat(y_samp, axis=-1)
            return y_pred
    
    def get_posterior(self, y, n_samples=10000):
        if np.ndim(y) == 2:
            y = np.expand_dims(y, axis=0)
        if not np.ndim(y) == 3:
            msg = 'The input data must be of shape {}.'.format(self.E1.input_shape)
            raise ValueError(msg)
        distE1params = self.E1(y)
        distE1 = self.E1mixture.get_distribution(distE1params)
        z = distE1.sample(n_samples)
        distDparams = self.D([np.repeat(y, n_samples, axis=0),
                              tf.squeeze(tf.concat(z, axis=-1))])
        distD = self.Dout.get_distribution(distDparams)
        out = tf.concat(distD.sample(), axis=-1)
        return np.array(out)
        
    def save(self, filepath, overwrite=True, options=None):
        os.makedirs(filepath, exist_ok=True)
        e1_path = os.path.join(filepath, 'E1')
        e2_path = os.path.join(filepath, 'E2')
        d_path = os.path.join(filepath, 'D')
        os.makedirs(e1_path, exist_ok=True)
        os.makedirs(e2_path, exist_ok=True)
        os.makedirs(d_path, exist_ok=True)
        
        #Save individual networks
        self.E1.save(e1_path, overwrite=overwrite, options=options)
        self.E2.save(e2_path, overwrite=overwrite, options=options)
        self.D.save(d_path, overwrite=overwrite, options=options)
        
        #Save output-distribution names
        with open(os.path.join(filepath, 'output_distributions.json'), 'w') as fp:
            json.dump(self.output_distributions, fp)
        
        #Save optimizer state
        opti_weights = self.optimizer.weights
        opti_config = self.optimizer.get_config()
        
        with open(os.path.join(filepath, 'optimizer_config.pkl'), 'wb') as fp:
            pickle.dump(opti_config, fp)
        
        order = []
        with h5py.File(os.path.join(filepath, 'optimizer_weights.hdf'), 'w') as fp:
            for weight in opti_weights:
                name = weight.name
                arr = weight.numpy()
                order.append(name)
                fp.create_dataset(name, data=arr)
            fp.create_dataset('order', data=np.array(order, dtype='S'))
    
    def set_ramp(self, ramp):
        self.ramp.assign(ramp)
        self.E1mixture.set_ramp(self.ramp)
        self.E2dist.set_ramp(self.ramp)
        self.Dout.set_ramp(self.ramp)
        

def load_cvae(filepath, dist_dict=None):
    e1_path = os.path.join(filepath, 'E1')
    e2_path = os.path.join(filepath, 'E2')
    d_path = os.path.join(filepath, 'D')
    
    #Load individual networks
    E1 = keras.models.load_model(e1_path)
    E2 = keras.models.load_model(e2_path)
    D = keras.models.load_model(d_path)
    
    #Load and create CVAE
    with open(os.path.join(filepath, 'output_distributions.json'), 'r') as fp:
        output_distributions = json.load(fp)
    ret = CVAE(E1, E2, D, output_distributions, dist_dict=dist_dict)
    
    #Load optimizer
    with open(os.path.join(filepath, 'optimizer_config.pkl'), 'rb') as fp:
        opti_config = pickle.load(fp)    
    opti = keras.optimizers.get({'class_name': opti_config['name'], 'config': opti_config})
    
    #Load optimizer weights
    with h5py.File(os.path.join(filepath, 'optimizer_weights.hdf'), 'r') as fp:
        order = fp['order'][()]
        opti_weights = []
        for key in order:
            opti_weights.append(fp[key][()])
    
    #Set optimizer weights
    with tf.name_scope(opti._name):
        with tf.init_scope():
            opti._create_all_weights(ret.trainable_variables)
    opti.set_weights(opti_weights)
    return ret
