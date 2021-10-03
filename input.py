import numpy
def dense_to_one_hot(labels_dense):
    num_labels = labels_dense.shape[0]
    num_class = 9
    labels_one_hot = numpy.zeros((num_labels, num_class), dtype=numpy.uint8)
    k = 0
    for label in labels_dense:
      labels_one_hot[k, int(label)] = 1
      k += 1
    return labels_one_hot
class DataSet(object):
  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0]
    self._num_examples = images.shape[0]
    images = images.astype(numpy.float32)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
def read_data_sets(images, labels, test_images, test_labels, one_hot = False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  TRAIN_SIZE = int(images.shape[0]*0.8)
  Train_Images = images
  if one_hot:
    Train_Labels = dense_to_one_hot(labels)
  else:
    Train_Labels = labels
  test_images = test_images
  if one_hot:
    test_labels = dense_to_one_hot(test_labels)
  else:
    test_labels = test_labels
  train_images = Train_Images[:TRAIN_SIZE]
  train_labels = Train_Labels[:TRAIN_SIZE]
  validation_images = Train_Images[TRAIN_SIZE:]
  validation_labels = Train_Labels[TRAIN_SIZE:]
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets
