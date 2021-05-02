
import numpy as np 
import pandas as pd
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import layers 
import tensorflow_probability as tfp 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from random import shuffle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import feather
import warnings
warnings.filterwarnings('ignore')
import tqdm

from tensorflow.keras.layers import Layer, InputSpec


class SOMLayer(Layer):
    """
    Self-Organizing Map layer class with rectangular topology
    # Example
    ```
        model.add(SOMLayer(map_size=(10,10)))
    ```
    # Arguments
        map_size: Tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1].
        prototypes: Numpy array with shape `(n_prototypes, latent_dim)` witch represents the initial cluster centers
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`
    # Output shape
        2D tensor with shape: `(n_samples, n_prototypes)`
    """

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.initial_prototypes = prototypes
        self.input_spec = InputSpec(ndim=2)
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_dim))
        self.prototypes = self.add_weight(shape=(self.n_prototypes, input_dim), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Calculate pairwise squared euclidean distances between inputs and prototype vectors
        Arguments:
            inputs: the variable containing data, Tensor with shape `(n_samples, latent_dim)`
        Return:
            d: distances between inputs and prototypes, Tensor with shape `(n_samples, n_prototypes)`
        """
        # Note: (tf.expand_dims(inputs, axis=1) - self.prototypes) has shape (n_samples, n_prototypes, latent_dim)
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.n_prototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

path='...'
df1 = pd.read_feather(path+'Brain_GSE15824.feather')
df2 = pd.read_feather(path+'Brain_GSE50161.feather')
df3 = pd.read_feather(path+'Breast_GSE45827.feather')
df4 = pd.read_feather(path+'Breast_GSE59246.feather')
df5 = pd.read_feather(path+'Leukemia_GSE28497.feather')
df6 = pd.read_feather(path+'Leukemia_GSE71935.feather')

print(df1.shape, df2.shape, df3.shape, 
      df4.shape, df5.shape, df6.shape)

df1.head()

X1 = df1.iloc[:,2:]
y1 = df1.iloc[:, 1:2]
X2 = df2.iloc[:,2:]
y2 = df2.iloc[:, 1:2]
X3 = df3.iloc[:,2:]
y3 = df3.iloc[:, 1:2]
X4 = df4.iloc[:,2:]
y4 = df4.iloc[:, 1:2]
X5 = df5.iloc[:,2:]
y5 = df5.iloc[:, 1:2]
X6 = df6.iloc[:,2:]
y6 = df6.iloc[:, 1:2]

print(y1.value_counts(), y2.value_counts(), y3.value_counts(), 
      y4.value_counts(), y5.value_counts(), y6.value_counts())

X1 = X1.values
y1 = y1.values
X2 = X2.values
y2 = y2.values
X3 = X3.values
y3 = y3.values
X4 = X4.values
y4 = y4.values
X5 = X5.values
y5 = y5.values
X6 = X6.values
y6 = y6.values


# ## K-Fold Traing & Validating

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=3, shuffle=True)
## BSOMs
cvscores1 = []
dataset_size = len(X1)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

lr = 0.005
epoch = 300

for train_idx, val_idx in kf.split(X1, y1):
    model_1 = keras.Sequential([
    keras.Input(shape=(X1.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(4, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
    ])
    
    model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y1 = OneHotEncoder().fit_transform(y1).toarray()


    model_1.fit(X1[train_idx], en_y1[train_idx],epochs=epoch, verbose=0)
    
    scores1 = model_1.evaluate(X1[val_idx], en_y1[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_1.metrics_names[1], scores1[1]*100))
    cvscores1.append(scores1[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)))

cvscores2 = []
dataset_size = len(X2)
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

lr = 0.005
epoch = 300

for train_idx, val_idx in kf.split(X2, y2):
    model_2 = keras.Sequential([
    keras.Input(shape=(X2.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(5, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
    ])
    
    model_2.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y2 = OneHotEncoder().fit_transform(y2).toarray()


    model_2.fit(X2[train_idx], en_y2[train_idx],epochs=epoch, verbose=0)
    
    scores2 = model_2.evaluate(X2[val_idx], en_y2[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_2.metrics_names[1], scores2[1]*100))
    cvscores2.append(scores2[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))

cvscores3 = []
dataset_size = len(X3)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

lr = 0.005
epoch = 300

for train_idx, val_idx in kf.split(X3, y3):
    model_3 = keras.Sequential([
    keras.Input(shape=(X3.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(16, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(6, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
    ])
    
    model_3.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y3 = OneHotEncoder().fit_transform(y3).toarray()


    model_3.fit(X3[train_idx], en_y3[train_idx],epochs=epoch, verbose=0)
    
    scores3 = model_3.evaluate(X3[val_idx], en_y3[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_3.metrics_names[1], scores3[1]*100))
    cvscores3.append(scores3[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores3), np.std(cvscores3)))

cvscores4 = []
dataset_size = len(X4)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

lr = 0.002
epoch = 300

for train_idx, val_idx in kf.split(X4, y4):
    model_4 = keras.Sequential([
    keras.Input(shape=(X4.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(16, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(2, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
    ])
    
    model_4.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y4 = OneHotEncoder().fit_transform(y4).toarray()


    model_4.fit(X4[train_idx], en_y4[train_idx],epochs=epoch, verbose=0)
    
    scores4 = model_4.evaluate(X4[val_idx], en_y4[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_4.metrics_names[1], scores4[1]*100))
    cvscores4.append(scores4[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores4), np.std(cvscores4)))

cvscores5 = []
dataset_size = len(X5)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

lr = 0.001
epoch = 300

for train_idx, val_idx in kf.split(X5, y5):
    model_5 = keras.Sequential([
    keras.Input(shape=(X5.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(16, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(7, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
    ])
    
    model_5.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y5 = OneHotEncoder().fit_transform(y5).toarray()


    model_5.fit(X5[train_idx], en_y5[train_idx],epochs=epoch, verbose=0)
    
    scores5 = model_5.evaluate(X5[val_idx], en_y5[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_5.metrics_names[1], scores5[1]*100))
    cvscores5.append(scores5[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores5), np.std(cvscores5)))

cvscores6 = []
dataset_size = len(X6)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

lr = 0.001
epoch = 300

for train_idx, val_idx in kf.split(X6, y6):
    model_6 = keras.Sequential([
    keras.Input(shape=(X6.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(2, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
    ])
    
    model_6.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y6 = OneHotEncoder().fit_transform(y6).toarray()


    model_6.fit(X6[train_idx], en_y6[train_idx],epochs=epoch, verbose=0)
    
    scores6 = model_6.evaluate(X6[val_idx], en_y6[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_6.metrics_names[1], scores6[1]*100))
    cvscores6.append(scores6[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores6), np.std(cvscores6)))


### DSOM

cvscores1 = []

lr = 0.002
epoch = 200

for train_idx, val_idx in kf.split(X1, y1):
    model_1 = keras.Sequential([
    keras.Input(shape=(X1.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.Dense(8, activation='relu',  name='Bdense'),
    layers.Dense(4, activation='softmax',name='out')
    ])
    
    model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y1 = OneHotEncoder().fit_transform(y1).toarray()


    model_1.fit(X1[train_idx], en_y1[train_idx],epochs=epoch, verbose=0)
    
    scores1 = model_1.evaluate(X1[val_idx], en_y1[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_1.metrics_names[1], scores1[1]*100))
    cvscores1.append(scores1[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)))

cvscores2 = []

lr = 0.001
epoch = 200

for train_idx, val_idx in kf.split(X2, y2):
    model_2 = keras.Sequential([
    keras.Input(shape=(X2.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.Dense(16, activation='relu',  name='Bdense'),
    layers.Dense(5, activation='softmax',name='out')
    ])
    
    model_2.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y2 = OneHotEncoder().fit_transform(y2).toarray()


    model_2.fit(X2[train_idx], en_y2[train_idx],epochs=epoch, verbose=0)
    
    scores2 = model_2.evaluate(X2[val_idx], en_y2[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_2.metrics_names[1], scores2[1]*100))
    cvscores2.append(scores2[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))

cvscores3 = []

lr = 0.005
epoch = 300

for train_idx, val_idx in kf.split(X3, y3):
    model_3 = keras.Sequential([
    keras.Input(shape=(X3.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.Dense(16, activation='relu',  name='Bdense'),
    layers.Dense(6, activation='softmax',name='out')
    ])
    
    model_3.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y3 = OneHotEncoder().fit_transform(y3).toarray()


    model_3.fit(X3[train_idx], en_y3[train_idx],epochs=epoch, verbose=0)
    
    scores3 = model_3.evaluate(X3[val_idx], en_y3[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_3.metrics_names[1], scores3[1]*100))
    cvscores3.append(scores3[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores3), np.std(cvscores3)))


cvscores4 = []

lr = 0.001
epoch = 200

for train_idx, val_idx in kf.split(X4, y4):
    model_4 = keras.Sequential([
    keras.Input(shape=(X4.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.Dense(16, activation='relu',  name='Bdense'),
    layers.Dense(2, activation='softmax',name='out')
    ])
    
    model_4.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y4 = OneHotEncoder().fit_transform(y4).toarray()


    model_4.fit(X4[train_idx], en_y4[train_idx],epochs=epoch, verbose=0)
    
    scores4 = model_4.evaluate(X4[val_idx], en_y4[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_4.metrics_names[1], scores4[1]*100))
    cvscores4.append(scores4[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores4), np.std(cvscores4)))

cvscores5 = []

lr = 0.001
epoch = 200

for train_idx, val_idx in kf.split(X5, y5):
    model_5 = keras.Sequential([
    keras.Input(shape=(X5.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.Dense(16, activation='relu',  name='Bdense'),
    layers.Dense(7, activation='softmax',name='out')
    ])
    
    model_5.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y5 = OneHotEncoder().fit_transform(y5).toarray()


    model_5.fit(X5[train_idx], en_y5[train_idx],epochs=epoch, verbose=0)
    
    scores5 = model_5.evaluate(X5[val_idx], en_y5[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_5.metrics_names[1], scores5[1]*100))
    cvscores5.append(scores5[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores5), np.std(cvscores5)))


cvscores6 = []

lr = 0.001
epoch = 200

for train_idx, val_idx in kf.split(X6, y6):
    model_6 = keras.Sequential([
    keras.Input(shape=(X6.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.Dense(8, activation='relu',  name='Bdense'),
    layers.Dense(2, activation='softmax',name='out')
    ])
    
    model_6.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    en_y6 = OneHotEncoder().fit_transform(y6).toarray()


    model_6.fit(X6[train_idx], en_y6[train_idx],epochs=epoch, verbose=0)
    
    scores6 = model_6.evaluate(X6[val_idx], en_y6[val_idx], verbose=0)
    print("%s: %.2f%%" %(model_6.metrics_names[1], scores6[1]*100))
    cvscores6.append(scores6[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores6), np.std(cvscores6)))

