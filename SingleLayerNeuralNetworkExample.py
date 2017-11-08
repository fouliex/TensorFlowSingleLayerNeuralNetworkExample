import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile


def download(url, file):
    """""
    Download file from <url>
    :param url: URL to file
    :param file:
    :return:
    """""
    if not os.path.isfile(file):
        print('Downloading' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')


def uncompress_features_labels(file):
    """""
    Uncompress features and label from a zip file
    :param file: The zip file to extract the data from
    :return:
    """""
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_progress_bar = tqdm(zipf.namelist(), unit='files')

        # Get features and labels from all files
        for filename in filenames_progress_bar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                if __name__ == '__main__':
                    with zipf.open(filename) as image_file:
                        image = Image.open(image_file)
                        image.load()
                        # Load image data as 1 dimensional array
                        # We're using float32 to save on memory space
                        feature = np.array(image, dtype=np.float32).flatten()
                    # Get the letter from the filename. This is the letter of the image
                    label = os.path.split(filename)[1][0]

                    features.append(feature)
                    labels.append(label)
        return np.array(features), np.array(labels)


def save_data(pickle_filename):
    pickle_file = pickle_filename
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_filename, 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': train_features,
                        'train_labels': train_labels,
                        'valid_dataset': valid_features,
                        'valid_labels': valid_labels,
                        'test_dataset': test_features,
                        'test_labels': test_labels,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to ', pickle_file, ':', e)
            raise
        print('Data cached in pickle file.')
        return pickle_file


def reload_data(pickle_filename, train_features, train_labels, valid_features, valid_labels, test_features,
                test_labels_):
    pickle_file = pickle_filename
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        train_features = pickle_data['train_dataset']
        train_labels = pickle_data['train_labels']
        valid_features = pickle_data['valid_dataset']
        valid_labels = pickle_data['valid_labels']
        test_features = pickle_data['test_dataset']
        test_labels = pickle_data['test_labels']
        del pickle_data  # Free up memory
    print('Data and Modules loaded.')

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


def normalize_grayscale(image_data):
    """""
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """""
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


if __name__ == '__main__':
    if __name__ == "__main__":
        # The notMNIST dataset is too large for many computers to handle. It contains 500000 images for just training.
        # In this example a subset of this data 15000 images for each level (A-J)
        download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
        download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

        # Make sure the files aren't corrupted
        assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa', \
            'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
        assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9', \
            'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

        print('All files downloaded.')

        # Get the features and labels from the zip files
        train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
        test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

        # Limit the amount of data to work with
        size_limit = 150000
        train_features, train_labels = resample(train_features, train_labels, n_samples=size_limit)

        # Set flags for feature engineering. This will prevent us from skipping an important step
        is_features_normal = False
        is_labels_encod = False

        print('All features and labels uncompressed.')

        if not is_features_normal:
            train_features = normalize_grayscale(train_features)
            test_features = normalize_grayscale(test_features)
            is_features_normal = True

        print('Features are normalize')

        if not is_labels_encod:
            # Turn labels into numbers and apply One-Hot Encoding
            encoder = LabelBinarizer()
            encoder.fit(train_labels)
            train_labels = encoder.transform(train_labels)
            test_labels = encoder.transform(test_labels)

            # Change to float32, so it can be multiplied against the features in  TensorFlow
            train_labels = train_labels.astype(np.float32)
            test_labels = test_labels.astype(np.float32)
            is_labels_encod = True

        print('Labels One-Hot Encoded')

        assert is_features_normal, 'You skipped the step to normalize the features'
        assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'

        # Get randomized datasets for training and validation
        train_features, valid_features, train_labels, valid_labels = train_test_split(train_features, train_labels,
                                                                                      test_size=0.05,
                                                                                      random_state=832289)

        print('Training features and labels randomized and split.')

        pickle_file = 'notMNIST.pickle'
        save_data(pickle_file)
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = reload_data(
            pickle_file, train_features, train_labels, valid_features, valid_labels, test_features, test_labels)

        features_count = 784
        labels_count = 10

        features = tf.placeholder(tf.float32)
        labels = tf.placeholder(tf.float32)

        # Set the features and labels tensors
        features = tf.placeholder(tf.float32)
        labels = tf.placeholder(tf.float32)

        # Set the weights and biases tensor
        weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
        biases = tf.Variable(tf.zeros(labels_count))

        # Linear Function WX + b
        logits = tf.matmul(features, weights) + biases

        prediction = tf.nn.softmax(logits)

        # Cross entropy
        cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

        # Training loss
        loss = tf.reduce_mean(cross_entropy)

        # Create an operation that initializes all variables
        init = tf.global_variables_initializer()

        # Determine if the predictions are correct
        is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

        # Calculate the accuracy of the predictions
        accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

        print('Accuracy function created')