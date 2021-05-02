import numpy as np 
import pandas as pd
import feather
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import layers 
import tensorflow_probability as tfp 
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from random import shuffle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import tqdm
import feather

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

seed= 0

X1_train, X1_test, y1_train, y1_test =   \
      train_test_split(X1, y1, test_size=0.1, stratify=y1, random_state=seed)

print(y1_train.value_counts())

X1_train = X1_train.values
y1_train = y1_train.values
X1_test = X1_test.values
y1_test = y1_test.values

X2_train, X2_test, y2_train, y2_test =    \
     train_test_split(X2, y2, test_size=0.1, stratify=y2, random_state=seed)

print(y2_train.value_counts())

X2_train = X2_train.values
y2_train = y2_train.values
X2_test = X2_test.values
y2_test = y2_test.values


X3_train, X3_test, y3_train, y3_test =     \
    train_test_split(X3, y3, test_size=0.1, stratify=y3, random_state=seed)

print(y3_train.value_counts())

X3_train = X3_train.values
y3_train = y3_train.values
X3_test = X3_test.values
y3_test = y3_test.values

X4_train, X4_test, y4_train, y4_test =     \
    train_test_split(X4, y4, test_size=0.1, stratify=y4, random_state=seed)

print(y4_train.value_counts())

X4_train = X4_train.values
y4_train = y4_train.values
X4_test = X4_test.values
y4_test = y4_test.values

X5_train, X5_test, y5_train, y5_test =     \
    train_test_split(X5, y5, test_size=0.1, stratify=y5, random_state=seed)

print(y5_train.value_counts())

X5_train = X5_train.values
y5_train = y5_train.values
X5_test = X5_test.values
y5_test = y5_test.values

X6_train, X6_test, y6_train, y6_test =     \
    train_test_split(X6, y6, test_size=0.1, stratify=y6, random_state=seed)

print(y6_train.value_counts())

X6_train = X6_train.values
y6_train = y6_train.values
X6_test = X6_test.values
y6_test = y6_test.values

# ## DSOMs
# ### M1

model1 = keras.Sequential([
    keras.Input(shape=(X1_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(3,3), name='som'),
    layers.Dense(8, activation='relu', name = 'dense'),
    layers.Dense(4, activation='softmax', name='out')
    ])

lr = 0.002
epoch = 200

model1.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y1_train = OneHotEncoder().fit_transform(y1_train).toarray()

history = model1.fit(X1_train, en_y1_train, verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

model1.evaluate(X1_test, OneHotEncoder().fit_transform(y1_test).toarray())

#  ### M2
model2 = keras.Sequential([
    keras.Input(shape=(X2_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.Dense(16, activation='relu', name = 'dense'),
    layers.Dense(5, activation='softmax', name='out')
    ])

lr = 0.001
epoch = 200

model2.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y2_train = OneHotEncoder().fit_transform(y2_train).toarray()

history2 = model2.fit(X2_train, en_y2_train,verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])

model2.evaluate(X2_test, OneHotEncoder().fit_transform(y2_test).toarray())


# ### M3
model3 = keras.Sequential([
    keras.Input(shape=(X3_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'), #SOMLayer(map_size=(2,3), name='som'),
    layers.Dense(16, activation='relu', name = 'dense'),
    layers.Dense(6, activation='softmax', name='out')
    ])

lr = 0.002
epoch = 300

model3.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y3_train = OneHotEncoder().fit_transform(y3_train).toarray()

history3 = model3.fit(X3_train, en_y3_train, verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])

model3.evaluate(X3_test, OneHotEncoder().fit_transform(y3_test).toarray())


# ### M4
model4 = keras.Sequential([
    keras.Input(shape=(X4_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.Dense(16, activation='relu', name = 'dense'),
    layers.Dense(2, activation='softmax', name='out')
    ])

lr = 0.001
epoch = 200

model4.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y4_train = OneHotEncoder().fit_transform(y4_train).toarray()

history4 = model4.fit(X4_train, en_y4_train, verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history4.history['accuracy'])
plt.plot(history4.history['val_accuracy'])

model4.evaluate(X4_test, OneHotEncoder().fit_transform(y4_test).toarray())


# ### M5
model5 = keras.Sequential([
    keras.Input(shape=(X5_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.Dense(16, activation='relu', name = 'dense'),
    layers.Dense(7, activation='softmax', name='out')
    ])

lr = 0.001
epoch = 200

model5.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y5_train = OneHotEncoder().fit_transform(y5_train).toarray()

history5 = model5.fit(X5_train, en_y5_train, verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history5.history['accuracy'])
plt.plot(history5.history['val_accuracy'])

model5.evaluate(X5_test, OneHotEncoder().fit_transform(y5_test).toarray())


# ### M6
model6 = keras.Sequential([
    keras.Input(shape=(X6_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(2,3), name='som'),
    layers.Dense(8, activation='relu', name = 'dense'),
    layers.Dense(2, activation='softmax', name='out')
    ])

lr = 0.001
epoch = 200

model6.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y6_train = OneHotEncoder().fit_transform(y6_train).toarray()

history6 = model6.fit(X6_train, en_y6_train, verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history6.history['accuracy'])
plt.plot(history6.history['val_accuracy'])

model6.evaluate(X6_test, OneHotEncoder().fit_transform(y6_test).toarray())


# ## BSOMs

# ### M1
dataset_size = len(X1_train)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

model_1 = keras.Sequential([
    keras.Input(shape=(X1_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(4, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
])

lr = 0.005
epoch = 300

model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y1_train = OneHotEncoder().fit_transform(y1_train).toarray()

history_1 = model_1.fit(X1_train, en_y1_train,verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])

BSOM1_preds=[]

for i in tqdm.tqdm(range(500)):
    y_p = model_1.predict(X1_test)
    BSOM1_preds.append(y_p)

## score of the BSOM1 model
accs1 = []
for y_p in BSOM1_preds:
    acc = accuracy_score(y1_test.argmax(axis=1), y_p.argmax(axis=1))
    accs1.append(acc)
    
print('BSOM1 accuracy: {:.1%}'.format(sum(accs1)/len(accs1)))

## explore predictions
idx = 0
p0 = np.array([p[idx] for p in BSOM1_preds])
print('posterior mean: {}'.format(p0.mean(axis=0).argmax()))
print('true label: {}'.format(y1_test[idx].argmax()))
print()
## probability + standard error
for i, (prob, sd) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print('class: {}; prob: {:.1%}; sd: {:.2%}'.format(i, prob, sd))
 
## visualize prediction details
x,y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x,y)

fig, axes = plt.subplots(2,2, figsize=(6,6))

for i,ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f'class {i}')
    ax.label_outer()


# ### M2
dataset_size = len(X2_train)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

model_2 = keras.Sequential([
    keras.Input(shape=(X2_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(5, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
])

lr = 0.005
epoch = 300

model_2.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y2_train = OneHotEncoder().fit_transform(y2_train).toarray()

history_2 = model_2.fit(X2_train, en_y2_train,verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])

BSOM2_preds=[]

for i in tqdm.tqdm(range(500)):
    y_p = model_2.predict(X2_test)
    BSOM2_preds.append(y_p)

## score of the BSOM1 model
accs2 = []
for y_p in BSOM2_preds:
    acc = accuracy_score(y2_test.argmax(axis=1), y_p.argmax(axis=1))
    accs2.append(acc)
    
print('BSOM2 accuracy: {:.1%}'.format(sum(accs2)/len(accs2)))

## explore predictions
idx = 3
p0 = np.array([p[idx] for p in BSOM2_preds])
print('posterior mean: {}'.format(p0.mean(axis=0).argmax()))
print('true label: {}'.format(y2_test[idx].argmax()))
print()
## probability + standard error
for i, (prob, sd) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print('class: {}; prob: {:.1%}; sd: {:.2%}'.format(i, prob, sd))

    
    
## visualize prediction details
x,y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x,y)

fig, axes = plt.subplots(3,2, figsize=(6,9))

for i,ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f'class {i}')
    ax.label_outer()


# ### M3
dataset_size = len(X3_train)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

model_3 = keras.Sequential([
    keras.Input(shape=(X3_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(16, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(6, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
])

lr = 0.005
epoch = 300

model_3.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y3_train = OneHotEncoder().fit_transform(y3_train).toarray()

history_3 = model_3.fit(X3_train, en_y3_train,verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])


BSOM3_preds=[]

for i in tqdm.tqdm(range(500)):
    y_p = model_3.predict(X3_test)
    BSOM3_preds.append(y_p)

## score of the BSOM1 model
accs3 = []
for y_p in BSOM3_preds:
    acc = accuracy_score(y3_test.argmax(axis=1), y_p.argmax(axis=1))
    accs3.append(acc)
    
print('BSOM3 accuracy: {:.1%}'.format(sum(accs3)/len(accs3)))


## explore predictions
idx = 3
p0 = np.array([p[idx] for p in BSOM3_preds])
print('posterior mean: {}'.format(p0.mean(axis=0).argmax()))
print('true label: {}'.format(y3_test[idx].argmax()))
print()
## probability + standard error
for i, (prob, sd) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print('class: {}; prob: {:.1%}; sd: {:.2%}'.format(i, prob, sd))

    
    
## visualize prediction details
x,y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x,y)

fig, axes = plt.subplots(3,2, figsize=(6,9))

for i,ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f'class {i}')
    ax.label_outer()


# ### M4
dataset_size = len(X4_train)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

model_4 = keras.Sequential([
    keras.Input(shape=(X4_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(2, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
])

lr = 0.002
epoch = 300

model_4.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y4_train = OneHotEncoder().fit_transform(y4_train).toarray()

history_4 = model_4.fit(X4_train, en_y4_train,verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history_4.history['accuracy'])
plt.plot(history_4.history['val_accuracy'])

BSOM4_preds=[]

for i in tqdm.tqdm(range(500)):
    y_p = model_4.predict(X4_test)
    BSOM4_preds.append(y_p)

## score of the BSOM1 model
accs4 = []
for y_p in BSOM4_preds:
    acc = accuracy_score(y4_test.argmax(axis=1), y_p.argmax(axis=1))
    accs4.append(acc)
    
print('BSOM4 accuracy: {:.1%}'.format(sum(accs4)/len(accs4)))


## explore predictions
idx = 0
p0 = np.array([p[idx] for p in BSOM4_preds])
print('posterior mean: {}'.format(p0.mean(axis=0).argmax()))
print('true label: {}'.format(y4_test[idx].argmax()))
print()
## probability + standard error
for i, (prob, sd) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print('class: {}; prob: {:.1%}; sd: {:.2%}'.format(i, prob, sd))

    
    
## visualize prediction details
x,y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x,y)


fig, axes = plt.subplots(2,1, figsize=(3,6))

for i,ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f'class {i}')
    ax.label_outer()


# ### M5
dataset_size = len(X5_train)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

model_5 = keras.Sequential([
    keras.Input(shape=(X5_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(5,5), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(16, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(7, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
])

lr = 0.001
epoch = 300

model_5.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y5_train = OneHotEncoder().fit_transform(y5_train).toarray()

history_5 = model_5.fit(X5_train, en_y5_train, verbose=0,
epochs=epoch, 
validation_split=0.1)

plt.plot(history_5.history['accuracy'])
plt.plot(history_5.history['val_accuracy'])

BSOM5_preds=[]

for i in tqdm.tqdm(range(500)):
    y_p = model_5.predict(X5_test)
    BSOM5_preds.append(y_p)

## score of the BSOM1 model
accs5 = []
for y_p in BSOM5_preds:
    acc = accuracy_score(y5_test.argmax(axis=1), y_p.argmax(axis=1))
    accs5.append(acc)
    
print('BSOM5 accuracy: {:.1%}'.format(sum(accs5)/len(accs5)))


## explore predictions
idx = 0
p0 = np.array([p[idx] for p in BSOM5_preds])
print('posterior mean: {}'.format(p0.mean(axis=0).argmax()))
print('true label: {}'.format(y5_test[idx].argmax()))
print()
## probability + standard error
for i, (prob, sd) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print('class: {}; prob: {:.1%}; sd: {:.2%}'.format(i, prob, sd))

    
    
## visualize prediction details
x,y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x,y)


fig, axes = plt.subplots(3,3, figsize=(9,9))

for i,ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f'class {i}')
    ax.label_outer()


# ### M6

dataset_size = len(X6_train)
dataset_size
dist = tfp.distributions
kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

model_6 = keras.Sequential([
    keras.Input(shape=(X6_train.shape[1], ), name='input'),
    layers.BatchNormalization(),
    SOMLayer(map_size=(4,4), name='som'),
    layers.BatchNormalization(),
    tfp.layers.DenseFlipout(8, activation='relu',kernel_divergence_fn=kl_divergence_function,  name='Bdense'),
    tfp.layers.DenseFlipout(2, activation='softmax', kernel_divergence_fn=kl_divergence_function, name='out')
])

lr = 0.001
epoch = 300

model_6.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

en_y6_train = OneHotEncoder().fit_transform(y6_train).toarray()

history_6 = model_6.fit(X6_train, en_y6_train, verbose=0, 
epochs=epoch, 
validation_split=0.1)

plt.plot(history_6.history['accuracy'])
plt.plot(history_6.history['val_accuracy'])

BSOM6_preds=[]

for i in tqdm.tqdm(range(500)):
    y_p = model_6.predict(X6_test)
    BSOM6_preds.append(y_p)

## score of the BSOM1 model
accs6 = []
for y_p in BSOM6_preds:
    acc = accuracy_score(y6_test.argmax(axis=1), y_p.argmax(axis=1))
    accs6.append(acc)
    
print('BSOM6 accuracy: {:.1%}'.format(sum(accs6)/len(accs6)))


## explore predictions
idx = 0
p0 = np.array([p[idx] for p in BSOM6_preds])
print('posterior mean: {}'.format(p0.mean(axis=0).argmax()))
print('true label: {}'.format(y6_test[idx].argmax()))
print()
## probability + standard error
for i, (prob, sd) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print('class: {}; prob: {:.1%}; sd: {:.2%}'.format(i, prob, sd))

    
    
## visualize prediction details
x,y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x,y)


fig, axes = plt.subplots(2,1, figsize=(3,6))

for i,ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f'class {i}')
    ax.label_outer()
