import os

os.environ['DDE_BACKEND'] = 'tensorflow'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import deepxde as dde
import numpy as np
import tensorflow as tf

dde.config.set_default_float("float64")
dde.config.set_random_seed(42)
np.random.seed(42)

from deepxde.data import PDE
from deepxde.geometry import TimeDomain
from deepxde.icbc.initial_conditions import IC
from deepxde.icbc import PointSetBC
from deepxde.nn import FNN, PFNN
from deepxde.model import Model
from deepxde.callbacks import VariableValue, EarlyStopping
import deepxde.backend as bkd

from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from .custom_models import AdaptativeDataWeightModel
from .data import transform_column_with_mask



class WorkflowModel:

    def __init__(
        self, 
        t_0, 
        t_f, 
        I_data, 
        data_t, 
        N=1, 
        gamma=0.1, 
        n_hidden_layers=3,
        hidden_layer_size=80,
        activation="tanh",
        learning_rate=0.002,
        scaling="z",
        adam_iterations=300000,
        lbfgs_iterations=50000, 
        adaptative_wdata=False,
        early_stopping=True,
        fine_tunning_using_lbfgs=False,
        beta_hard_constraints=False
    ):
        self.t_0 = t_0
        self.t_f = t_f
        self.I_data = I_data
        self.data_t = data_t.reshape(-1, 1)
        self.N = N
        self.gamma = gamma
        self.n_equations = 2
        self.n_compartments = 2
        self.n_out = self.n_compartments + 1
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.adam_iterations = adam_iterations
        self.lbfgs_iterations = lbfgs_iterations
        self.scaling = scaling
        self.adaptative_wdata = adaptative_wdata
        self.early_stopping = early_stopping
        self.fine_tunning_using_lbfgs = fine_tunning_using_lbfgs
        self.beta_hard_constraints = beta_hard_constraints
        self.scale_data()
        self.config_model()


    def scale_data(self):

        match self.scaling:
            case  "z":
                I_mean = self.I_data.mean(axis=0)
                I_std = self.I_data.std(axis=0)
                def scale(data): return (data - I_mean) / I_std
                def unscale(data): return data * I_std + I_mean
            case "min/max":
                I_min = self.I_data.min(axis=0)
                I_max = self.I_data.max(axis=0)
                def scale(data): return (data - I_min) / (I_max - I_min)
                def unscale(data): return I_min + (I_max - I_min) * data
            case "norm":
                def scale(data): return data / self.N
                def unscale(data): return data * self.N
            case _:
                def scale(data): return data
                def unscale(data): return data

        self.scale = scale
        self.unscale = unscale

        self.scaled_I_data = scale(self.I_data)


    def create_ics(self):
        self.I0 = self.scaled_I_data[0]
        self.scaled_N = self.scale(self.N)
        self.S0 = self.scaled_N - self.I0
        self.beta0 = 0.25

        # Tensorflow has an issue with lambdas...
        def is_on_initial(_, on_initial): return on_initial
        def S0_val(_): return self.S0
        def I0_val(_): return self.I0
        def beta_val(_): return self.beta0

        return [
            IC(self.timeinterval, S0_val, is_on_initial, component=0),
            IC(self.timeinterval, I0_val, is_on_initial, component=1),
            # IC(self.timeinterval, beta_val, is_on_initial, component=2),
        ]


    def create_data_bcs(self):
        data_I = PointSetBC(self.data_t, self.scaled_I_data.reshape(-1, 1), component=1)
        return [data_I]


    def config_model(self):

        self.timeinterval = TimeDomain(self.t_0, self.t_f)

        def sir_residual(t, y):
            S, I, beta = y[:,0:1], y[:,1:2], y[:,2:3]

            dS_dt = dde.gradients.jacobian(y, t, i=0)
            dI_dt = dde.gradients.jacobian(y, t, i=1)

            return [
                dS_dt + beta * S * I / self.scaled_N,
                dI_dt - beta * S * I / self.scaled_N + self.gamma * I,
            ]

        ics = self.create_ics()
        dcs = self.create_data_bcs()

        data = PDE(
            self.timeinterval,
            sir_residual,
            ics + dcs,
            num_domain=len(self.data_t)*2,
            num_boundary=2,
            num_test=len(self.data_t)//2,
            anchors=self.data_t
        )

        topology = [1] + [self.hidden_layer_size] * self.n_hidden_layers + [self.n_out]

        initialization = "Glorot uniform" if self.activation == "tanh" else "He uniform"

        net = PFNN(
            topology,
            self.activation,
            initialization
        )

        if self.beta_hard_constraints:
            def non_negative(x, y):
                return transform_column_with_mask(y, 2, tf.nn.relu)
            net.apply_output_transform(non_negative)

        if self.adaptative_wdata:
            self.model = AdaptativeDataWeightModel(
                data, net, n_physics=self.n_equations + len(ics), n_data=1)
        else:
            self.model = Model(data, net)

        eq_w, ic_w, data_w = 1, 1, 1
        loss_weights = [eq_w] * self.n_equations + [ic_w] * len(ics) + [data_w] * len(dcs)

        self.model.compile("adam", self.learning_rate, loss_weights=loss_weights)


    def train(self, verbose=0):
        
        callbacks = []

        if self.early_stopping:
            callbacks.append(EarlyStopping(min_delta=1e-13, patience=15000))

        losshistory, train_state = self.model.train(
            iterations=self.adam_iterations,
            display_every=100,
            callbacks=callbacks,
            verbose=verbose
        )

        if self.fine_tunning_using_lbfgs:
            self.model.compile("L-BGFS")
            losshistory, train_state =  self.model.train(
                iterations=self.lbfgs_iterations,
                display_every=100,
                callbacks=callbacks
            )

        self.losshistory = losshistory
        self.train_state = train_state

        return losshistory, train_state


    def predict(self, t):
        pred = self.model.predict(t.reshape(-1, 1))
        comparts = self.unscale(pred[:,0:self.n_compartments])
        return np.vstack((comparts.T, pred[:,self.n_compartments])).T


    @property
    def data_weight_hist(self):
        if isinstance(self.model, AdaptativeDataWeightModel): 
            return self.model.data_weight_hist
        return None