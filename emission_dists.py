import sys
import numpy as np
import edflib
from dataio import load_signals, get_stages_array
from epoch import get_epoch_data, number_of_epochs
from features import get_features



def compute_emission_distribution(data_file, label_file):

    e = edflib.EdfReader(data_file)

    # Signal numbers
    # 5 -> EOG Left (freq: 50Hz)
    # 7 -> EEG_1 (freq: 125Hz)
    # 8 -> Respiration (freq: 10Hz)

    signal_indices = [4, 7, 8]
    # Signal frequencies
    eogl_f = 125; eeg1_f = 125; resp_f = 10;
    eogl, eeg1, resp = load_signals(e, signal_indices)
    num_epochs = number_of_epochs(eeg1, eeg1_f)

    print "Number of epochs: %d" %num_epochs
    num_epochs_verify = number_of_epochs(eogl, eogl_f)
    assert num_epochs == num_epochs_verify

    # 2) Load epoch labels from labeled data
    stages = get_stages_array(label_file)

    assert len(stages) == num_epochs

    # 3) Create feature matrices for each hidden state
    first_w, first_n, first_r = [True for i in range(0, 3)]

    # For each epoch in the signal (ei: epoch index)
    for ei in xrange(0, num_epochs):
        # Retrieve data in ei-th epoch
        e1  = get_epoch_data(eeg1, eeg1_f, ei, 1, mute)     # EEG 1
        r   = get_epoch_data(resp, resp_f, ei, 1, mute)     # Respiration
        eol  = get_epoch_data(eogl, eogl_f, ei, 1, mute)    # EOG Left

        # Data label (Wake, NREM, REM)
        label = stages[ei]

        # Extract the features from the data
        e1_features = get_features(e1, eeg1_f)
        r_features = get_features(r, resp_f)
        eogl_features = get_features(eol, eogl_f)

        # Features is features vector composed of features of many signals stacked together
        features = np.hstack((e1_features, r_features, eogl_features))

        if label == 'W':
            if mute is False: print "Epoch has label [Wake]"
            if first_w is True:
                w_feats = features
                first_w = False
            elif first_w is False:
                w_feats = np.column_stack((w_feats, features))

        if label == "N":
            if mute is False: print "Epoch has label [NREM]"
            if first_n is True:
                n_feats = features
                first_n = False
            elif first_n is False:
                n_feats = np.column_stack((n_feats, features))

        if label == "R":
            if mute is False: print "Epoch has label [REM]"
            if first_r is True:
                r_feats = features
                first_r = False
            elif first_r is False:
                r_feats = np.column_stack((r_feats, features))

    # print w_feats; print n_feats; print r_feats