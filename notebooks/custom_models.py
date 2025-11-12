import deepxde as dde
import deepxde.config as config
import deepxde.optimizers as optimizers
import tensorflow as tf
import sys
from deepxde.callbacks import Callback


class AdaptativeDataWeightModel(dde.Model):

        

    def __init__(
        self, 
        net, 
        data, 
        n_physics=None, 
        n_data=None, 
        momentum_beta=0.9, 
        data_weight=1
    ):
        super().__init__(net, data)
        self.n_physics = n_physics
        self.n_data = n_data
        self.momentum_beta = momentum_beta
        self.data_weight = data_weight
        self.data_weight_hist = []



    def _compile_tensorflow(self, lr, loss_fn, decay):
        """tensorflow"""

        super()._compile_tensorflow(lr, loss_fn, decay)
        
        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)
        
        # Had to disable earger execution...
        # @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
                losses = self.outputs_losses_train(inputs, targets, auxiliary_vars)[1]

                phys_losses = losses[:self.n_physics]
                data_losses = losses[self.n_physics:self.n_physics + self.n_data]
                    
                total_loss = tf.math.reduce_sum(losses)

                phys_loss = tf.math.reduce_sum(phys_losses)
                data_loss = tf.math.reduce_sum(data_losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )

            phys_grads = tape1.gradient(phys_loss, trainable_variables)
            data_grads = tape2.gradient(data_loss, trainable_variables)

            grads = tape3.gradient(total_loss, trainable_variables)

            max_phys_grad_norm = tf.math.reduce_max(tf.stack([tf.norm(g) for g in phys_grads if g is not None]))
            mean_data_grad_norm = tf.math.reduce_mean(tf.stack([tf.norm(g) for g in data_grads if g is not None]))

            lambda_ratio = tf.math.exp(tf.math.log(max_phys_grad_norm) - tf.math.log(mean_data_grad_norm))

            new_data_weight = self.momentum_beta * self.data_weight + (1 - self.momentum_beta) * lambda_ratio
            self.data_weight = new_data_weight.numpy()
            
            # print(self.loss_weights)
            self.data_weight_hist.append(self.data_weight)
            self.loss_weights[self.n_physics:self.n_physics + self.n_data] = [self.data_weight] * self.n_data   
            
            opt.apply_gradients(zip(grads, trainable_variables))

        self.train_step = train_step
       