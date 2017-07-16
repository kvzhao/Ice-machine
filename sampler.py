import numpy as np
import h5py
import os, sys
import collections

from scipy.misc import imsave

from random import shuffle

from constants import DATASET_PATH
from constants import DATASET_NAME
from constants import IMAGE_NAME

def read_data_sets(grayscale=False, reshape=False):
    h = h5py.File('/'.join([DATASET_PATH, DATASET_NAME]), 'r+')
    train_images = h[IMAGE_NAME][:]
    train_labels = np.ones_like(train_images)
    h.close()

    data = DataSet(images=train_images, labels=train_labels, grayscale=grayscale, reshape=reshape)
    return data

class NoiseSampler(object):
    def __call__ (self, batch_size, z_dim):
        return np.random.uniform(-1.0, +1.0, [batch_size, z_dim])

class DataSampler(object):
    def __init__ (self):
        self.shape = [64, 64, 1]
        #self.dim = 64 * 64 * 1
        self.dataset = read_data_sets(grayscale=False, reshape=True)
    
    def __call__(self, batch_size):
        return self.dataset.next_batch(batch_size)[0]

class DataSet(object):
	def __init__ (self,
                      images,
                      labels,
                      dtype=np.float32,
                      grayscale=True,
                      reshape=True):
            self._images = images
            self._labels = labels
            self._num_of_samples = images.shape[0]
            self._epochs_completed = 0
            self._index_in_epoch = 0
            #self._dim = 64 * 64 * 1
            if (reshape):
                self._images = self._images[..., np.newaxis]
                #self._images = self._images.reshape((self._num_of_samples, self._dim))
            if(grayscale):
                self._convert_to_grayscale()

        @property
        def images(self):
            return self._images

        @property
        def labels(self):
            return self._labels

        @property
        def num_samples(self):
            return self._num_of_samples

        @property
        def epochs_completed(self):
            return self._epochs_completed

	def _convert_to_grayscale(self):
            for i in range(self._num_of_samples):
                c = self._images[i]
                c[c <= 0.0] = 0.0
                self._images[i] = c
            self._images = self._images.astype(np.uint8)
            #  And shuffl the data
            perm = np.arange(self._num_of_samples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]

	def next_batch(self, batch_size, shuffle=True):
		start = self._index_in_epoch
		#self._index_in_epoch += batch_size

                # Shuffle the first epoch
                if self._epochs_completed == 0 and start == 0 and shuffle:
                    perm0 = np.arange(self._num_of_samples)
                    np.random.shuffle(perm0)
                    self._images = self.images[perm0]
                    self._labels = self.labels[perm0]
                # Finsh the epoch
                if start + batch_size > self._num_of_samples:
                    self._epochs_completed += 1
                    rest_num_samples = self._num_of_samples - start
                    images_rest_part = self._images[start:self._num_of_samples]
                    labels_rest_part = self._labels[start:self._num_of_samples]
                     # Shuffle
                    if shuffle:
                        perm = np.arange(self._num_of_samples)
                        np.random.shuffle(perm)
                        self._images = self.images[perm]
                        self._labels = self.labels[perm]
                    # Start next epoch
                    start = 0
                    self._index_in_epoch = batch_size - rest_num_samples
                    end = self._index_in_epoch
                    images_new_part = self._images[start:end]
                    labels_new_part = self._labels[start:end]
                    return np.concatenate((images_rest_part, images_new_part), axis=0), \
                            np.concatenate((labels_rest_part, labels_new_part), axis=0)
                else:
                    self._index_in_epoch += batch_size
                    end = self._index_in_epoch
                    return self._images[start:end], self._labels[start:end]


