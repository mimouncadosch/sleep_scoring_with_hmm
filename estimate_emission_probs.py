import sys
import numpy as np
import edflib
from dataio import load_signals, get_stages_array, get_signal_frequencies, get_signals
from epoch import get_epoch_data, number_of_epochs
from features import get_features
from estimate_transition_probs import parse_filename

def estimate_emission_probs(train_data_file, label_file, mute=False):
    """
    Estimate the emission probabilities of the hidden states.
    It assumes that each hidden state produces observations following a Gaussian distribution.
    :param train_data_file: the file with the training data (.edf)
    :param label_file: the file with labels for the training data (.csv)
    :param mute: do not print some messages to screen

    """
    print "Estimating emission probabilities for file %s" %train_data_file

    # 1) Read EDF file and setup

    _, _, session_num, _ = parse_filename(label_file)

    # EDF reader object
    e = edflib.EdfReader(train_data_file)

    # Signal numbers
    # 5 -> EOG Left (freq: 50Hz)
    # 7 -> EEG_1 (freq: 125Hz)
    # 8 -> Respiration (freq: 10Hz)

    signal_indices = [5,7]
    # Signal frequencies
    freqs = get_signal_frequencies(e, signal_indices)
    signals = get_signals(e, signal_indices)
    print "Signal frequencies: ", freqs

    num_epochs = number_of_epochs(signals[0], freqs[0])
    print "Number of epochs: %d" %num_epochs


    # Verify length of signals are consistent
    if len(signal_indices) > 1:
        num_epochs_verify = number_of_epochs(signals[1], freqs[1])
        assert num_epochs == num_epochs_verify

    # 2) Load epoch labels from labeled data
    stages = get_stages_array(label_file)
    assert len(stages) == num_epochs

    # 3) Create feature matrices for each hidden state
    first_w, first_n, first_r = [True for i in range(0, 3)]

    # For each epoch in the signal (ei: epoch index)
    for ei in xrange(0, num_epochs):

        # Data label (Wake, NREM, REM)
        label = stages[ei]

        # For each signal
        first_sig = True
        for sid in range(0, len(signal_indices)):
            # Retrieve data in ei-th epoch
            epoch_data = get_epoch_data( signals[sid], freqs[sid], ei, 1, mute )
            # Extract the features from the data
            epoch_feats = get_features( epoch_data, freqs[sid])

            # Features is features vector composed of features of many signals stacked together
            if first_sig is True:
                features = epoch_feats
                first_sig = False
            else:
                features = np.hstack( (features, epoch_feats) )

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

    # 4) Compute mean vectors, and covariance matrices for each hidden state
    # For feature matrix of each hidden state, compute average of all features across all observations
    mu_w = w_feats.mean(1); mu_n = n_feats.mean(1); mu_r = r_feats.mean(1)

    # For feature matrix of each hidden state, compute covariance matrix.
    # Use np.cov, documentation: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # np.cov(X), where X s.t.: Each row of m represents a variable, and each column a single observation of all those variables.
    sigma_w = np.cov(w_feats); sigma_n = np.cov(n_feats); sigma_r = np.cov(r_feats)

    print "Number of features: %d" %len(mu_w)
    print "Saving parameters (mu and sigma) to './data/params/%s.npz" %session_num
    np.savez('./data/params/' + session_num, mu_w=mu_w, mu_n=mu_n, mu_r=mu_r, sigma_w=sigma_w, sigma_r=sigma_r, sigma_n=sigma_n)

    del e
    return True


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Wrong number of arguments."
        print "Usage: python estimate_emission_probs.py <train_data_file.edf> <reduced_label_file.csv>"
        print "Example: python estimate_emission_probs.py ./data/edfs/shhs1-200002.edf ./data/annotations/shhs1-200002-staging-reduced.csv"

        sys.exit()

    train_data_file = sys.argv[1]
    reduced_label_file = sys.argv[2]

    estimate_emission_probs(train_data_file, reduced_label_file)
