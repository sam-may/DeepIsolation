import os
import tensorflow as tf

import numpy
import sys
import h5py

import model_tpu

# Constants
nEpochs = 100
nBatch = 8192
nTrain = nBatch * 2

# Read features from hdf5 file
f = h5py.File("test/features.hdf5", "r")

global_features = f['global']
charged_pf_features = f['charged_pf']
photon_pf_features = f['photon_pf']
neutralHad_pf_features = f['neutralHad_pf']
#outer_pf_features = f['outer_pf']
label = f['label']
relIso = f['relIso']

#global_features = numpy.transpose(numpy.array([relIso])) # uncomment this line to train with only pf cands + RelIso

n_global_features = len(global_features[0])
n_charged_pf_features = len(charged_pf_features[0][0])
n_photon_pf_features = len(photon_pf_features[0][0])
n_neutralHad_pf_features = len(neutralHad_pf_features[0][0])

charged_pf_timestep = len(charged_pf_features[0])
photon_pf_timestep = len(photon_pf_features[0])
neutralHad_pf_timestep = len(neutralHad_pf_features[0])

print(n_global_features)
print(n_charged_pf_features)
print(n_photon_pf_features)
print(n_neutralHad_pf_features)
print(len(label))

print(charged_pf_timestep)
print(photon_pf_timestep)
print(neutralHad_pf_timestep)

################
# Structure NN #
################

scale = float(sys.argv[1])

model = model_tpu.parallel(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features, scale)

use_tpu = False
if use_tpu:
	strategy = tf.contrib.tpu.TPUDistributionStrategy(
		tf.contrib.cluster_resolver.TPUClusterResolver(tpu='osg01')
		)

	model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer = optimizer, loss = tf.keras.losses.binary_crossentropy, metrics = ['accuracy'])

print(model.summary())

model.fit([charged_pf_features[:nTrain], photon_pf_features[:nTrain], neutralHad_pf_features[:nTrain], global_features[:nTrain]], label[:nTrain], epochs = nEpochs, batch_size = nBatch)

