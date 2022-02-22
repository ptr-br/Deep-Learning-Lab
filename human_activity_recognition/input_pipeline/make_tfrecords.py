import numpy as np
import gin
import logging
import pandas as pd
import os
import sys
import tensorflow as tf
from scipy.stats import zscore, mode
from sklearn.utils import shuffle
from scipy.signal import butter, filtfilt


def create_windows_df(df, window_length, window_shift):
    """ 
    Create pieces of data that can be feed to the model later

    Args:
        df (dataframe): dataframe containing all the sensor data and labels from a file
        window_length (int): length of the window
        window_shift (int): shift of each window

    Returns:
        features, labels: array of features and correspondinng labels (or one label) 
        """

    features_list = []
    labels_list = []

    # delete first 5s and last 5s data. set first index to window_length
    for index in range(window_length, len(df) - window_length, window_shift):
        window_data = df.iloc[index : (index + window_length)].values

        # assign single label or complete sequence

        # - assign transition activity label if more then 40% of the sequence are transition
        # - assign normal activity label if more then 85% of the sequence are normal
        # -> assign zero else (get removed later)
        label, count = mode(window_data[:, 6]).mode[0], mode(window_data[:, 6]).count[0]

        if label >= 7:
            if (count / window_length) <= 0.4:
                label = 0
        else:
            if (count / window_length) <= 0.85:
                label = 0

        # assign features
        features = window_data[:, :-1]

        features_list.append(features)
        labels_list.append(label)

    labels_list = np.expand_dims(np.array(labels_list), axis=1)
    features_list = np.array(features_list)
    # shape of feature (w_nums, w_length, 6)  label(w_nums, )
    return features_list, labels_list


def low_pass(signal, fs=50, fc=10):
    """
    Lowpass filter signal

    Args:
        signal (np.array): input signal as pandas df
        fs (int, optional): sampling frequency of the signal. Defaults to 50.
        fc (int, optional): cutoff frequency. Defaults to 10.
    """

    w = fc / (fs / 2)
    b, a = butter(5, w, "low")
    signal_filtered = filtfilt(b, a, signal.T)
    return signal_filtered.T


def serialize_data(features, labels):
    """
    Create a tf.train.Example ready to be written to file.

    Args:
        features (tensor): features as tensor
        labels (tensor): labels as tensor

    """

    features = tf.io.serialize_tensor(features)
    labels = tf.io.serialize_tensor(labels)

    feature_of_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[features.numpy()]))

    labels_of_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.numpy()]))

    store_dict = {"features": feature_of_bytes, "labels": labels_of_bytes}

    example_proto = tf.train.Example(features=tf.train.Features(feature=store_dict))

    return example_proto.SerializeToString()


def record_writer(features, labels, filepath):
    """
    Write data to records files

    Args:
        features (np.array): features (train|test|val)
        labels ([type]): labels (train|test|val)
        filepath (str): path to store the records file
    """

    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    with tf.io.TFRecordWriter(filepath) as writer:
        for feature, label in ds:
            example = serialize_data(feature, label)
            writer.write(example)


def resample_data(features, labels):
    """
    Ensure that train dataset is balanced
    Args:
        labels (np.array): labels of the train dataset
        features (np.array): features of the train dataset

    Returns:
        features, labels: balanced dataset
    """

    features_resampled = np.empty((0, features.shape[1], features.shape[2]))
    labels_resampeld = np.empty((0, labels.shape[1]))

    activities, activity_counts = np.unique(labels, return_counts=True)

    # get max counts for both transition and static/dynamic activities
    max_act = np.max(activity_counts[1:7])
    max_transition_act = np.max(activity_counts[7:])

    # get indices if each activity and resample it to the max size
    # remove zero from indexes
    for activity in activities:
        activity_indices = np.where(labels == activity)[0]
        if activity < 7:
            indices = np.random.choice(activity_indices, size=max_act, replace=True)
        else:
            indices = np.random.choice(activity_indices, size=max_transition_act, replace=True)

        labels_resampeld = np.append(labels_resampeld, labels[indices], axis=0)
        features_resampled = np.append(features_resampled, features[indices], axis=0)

    return features_resampled, labels_resampeld


@gin.configurable
def create_tfrecords(data_dir, records_dir, window_length_and_shift, fc=False):
    """Creates tf-records files

    Args:
        data_dir (str): path to data 
        records_dir (str): path where tf-record files should be stored
        window_length_and_shift (tuple): set the size and shift of the window (size, shift)
        fc (int, optional): cutoff frequency. Defaults to False (no execution).
        
    Returns:
        bool: tells main script if tf-record files already exist
    """

    window_length, window_shift = window_length_and_shift

    # get speciic dir for each window_length & window_shift combination
    if fc > 0 and fc < 25:
        records_dir = records_dir + f"/wl{str(window_length)}_ws{str(window_shift)}_fc{str(fc)}"
    else:
        records_dir = records_dir + f"/wl{str(window_length)}_ws{str(window_shift)}"
        
        
    # if path already exists don't create files
    if os.path.exists(records_dir):
        return False

    features_train = np.empty(shape=(0, window_length, 6))
    features_val = np.empty(shape=(0, window_length, 6))
    features_test = np.empty(shape=(0, window_length, 6))

    labels_train = np.empty(shape=(0, 1))
    labels_val = np.empty(shape=(0, 1))
    labels_test = np.empty(shape=(0, 1))

    # TODO: CHECK IF DIR ALREADY EXISTS...

    if not os.path.exists(os.path.join(data_dir, "labels.txt")):
        logging.error("Path does not exist:")
        logging.error(os.path.join(data_dir, "labels.txt"))
        logging.error("Please provide correct path!")
        logging.error("Exiting script now! Bye.")
        sys.exit()

    logging.info("Creating tf-record files...")

    labels = pd.read_csv(os.path.join(data_dir, "labels.txt"), sep=" ", header=None)

    # LABEL-STRUCTURE:
    # column1 -> exp: experiment id
    # column2 -> usr: user id (1-30)
    # column3 -> act: activity id (0-12 => activity_labels.txt provides legend )
    # column4 -> sco: start count (in corresponding log file)
    # column5 -> eco: end count (in corresponding log file)

    # intital values for some params
    current_exp = -1
    current_usr = -1
    labels_length = len(labels)

    for index, (exp, usr, act, sco, eco) in labels.iterrows():

        # only load a new file if needed, otherwise keep content from already opened file
        if (exp != current_exp) or (usr != current_usr):

            # update indexes
            current_exp = exp
            current_usr = usr

            # read files data to dataframe
            acc_data = pd.read_csv(
                os.path.join(data_dir, f"acc_exp{str(current_exp).zfill(2)}_user{str(current_usr).zfill(2)}.txt"),
                sep=" ",
                header=None,
            )
            gyro_data = pd.read_csv(
                os.path.join(data_dir, f"gyro_exp{str(current_exp).zfill(2)}_user{str(current_usr).zfill(2)}.txt"),
                sep=" ",
                header=None,
            )
            
            sensor_data = pd.concat([acc_data, gyro_data], axis=1)
            
            # low pass filter if wanted the data
            if fc > 0 and fc < 25:
                sensor_data = low_pass(sensor_data, fc=fc)
                # make new data frame since lowpass filtering removes it 
                sensor_data = pd.DataFrame(sensor_data)
            
            sensor_data.columns = ["acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"]

            # normalize
            norm_sensor_data = zscore(sensor_data, axis=0)

            # set initial label
            norm_sensor_data["label"] = 0

        norm_sensor_data.loc[sco:eco, "label"] = act

        # only update if we need to load a new file afterwards
        if (index + 1) < labels_length:
            if (labels.iloc[index + 1, 0] != current_exp) or (labels.iloc[index + 1, 1] != current_usr):

                windowed_features, windowed_labels = create_windows_df(norm_sensor_data, window_length, window_shift)

                # NOTE: split ranges come from assignment description
                # train -> (1-21)
                # test  -> (22-27)
                # val   -> (28-30)

                if current_usr in range(1, 22):
                    features_train = np.append(features_train, windowed_features, axis=0)
                    labels_train = np.append(labels_train, windowed_labels, axis=0)
                elif current_usr in range(22, 28):
                    features_test = np.append(features_test, windowed_features, axis=0)
                    labels_test = np.append(labels_test, windowed_labels, axis=0)
                elif current_usr in range(28, 31):
                    features_val = np.append(features_val, windowed_features, axis=0)
                    labels_val = np.append(labels_val, windowed_labels, axis=0)

        else:

            windowed_features, windowed_labels = create_windows_df(norm_sensor_data, window_length, window_shift)

            if current_usr in range(1, 22):
                features_train = np.append(features_train, windowed_features, axis=0)
                labels_train = np.append(labels_train, windowed_labels, axis=0)
            elif current_usr in range(22, 28):
                features_test = np.append(features_test, windowed_features, axis=0)
                labels_test = np.append(labels_test, windowed_labels, axis=0)
            elif current_usr in range(28, 31):
                features_val = np.append(features_val, windowed_features, axis=0)
                labels_val = np.append(labels_val, windowed_labels, axis=0)

    # delete samples that do not have an activity label
    no_activity_indices_train = np.where(labels_train == 0)[0]
    no_activity_indices_test = np.where(labels_test == 0)[0]
    no_activity_indices_val = np.where(labels_val == 0)[0]

    labels_train = np.delete(labels_train, no_activity_indices_train, axis=0)
    labels_test = np.delete(labels_test, no_activity_indices_test, axis=0)
    labels_val = np.delete(labels_val, no_activity_indices_val, axis=0)

    features_train = np.delete(features_train, no_activity_indices_train, axis=0)
    features_test = np.delete(features_test, no_activity_indices_test, axis=0)
    features_val = np.delete(features_val, no_activity_indices_val, axis=0)

    # resample training dataset
    features_train, labels_train = resample_data(features_train, labels_train)

    # shuffle dataset (sequences that belong together should not be next to each other)
    features_train, labels_train = shuffle(features_train, labels_train)
    features_test, labels_test = shuffle(features_test, labels_test)
    features_val, labels_val = shuffle(features_val, labels_val)

    # logging number of samplesn
    logging.info("")
    logging.info(f"training samples:    {features_train.shape[0]}")
    logging.info(f"validation samples:  {features_val.shape[0]}")
    logging.info(f"test samples:        {features_test.shape[0]}")
    logging.info("")

    # create path and write record files
    os.makedirs(records_dir)

    record_writer(features_train, labels_train, records_dir + "/train.tfrecords")
    record_writer(features_val, labels_val, records_dir + "/validation.tfrecords")
    record_writer(features_test, labels_test, records_dir + "/test.tfrecords")

    return True
