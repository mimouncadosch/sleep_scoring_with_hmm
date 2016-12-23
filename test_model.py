import os, sys
import edflib
import numpy as np
from dataio import get_stages_array, get_signal_frequencies, get_signals
from epoch import get_epoch_data, number_of_epochs
from features import get_features

def predict_labels(test_data_file, model_session_num, signal_indices, mute):

    # 1) Get data
    e = edflib.EdfReader(test_data_file)

    freqs = get_signal_frequencies(e, signal_indices)
    signals = get_signals(e, signal_indices)

    print "Signal frequencies: ", freqs

    num_epochs = number_of_epochs(signals[0], freqs[0])

    first_feat = True
    # Retrieve data in ei-th epoch
    for ei in xrange(0, num_epochs):

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

        if first_feat is True:
            feat_mat = features
            first_feat = False
        else:
            feat_mat = np.vstack( (feat_mat, features) )

    # 2) Construct and train model
    from hmmlearn import hmm

    trans_mat_file = './data/transition_matrices/' + model_session_num + '_transitions.npy'
    print "Loading transition matrix %s" %trans_mat_file

    params = np.load('./data/params/' + model_session_num + '.npz')
    print "Loading emission parameters from ./data/params/%s.npz" %model_session_num

    mu_w = params['mu_w']; mu_n = params['mu_n']; mu_r = params['mu_r'];
    sigma_w = params['sigma_w']; sigma_n = params['sigma_n']; sigma_r = params['sigma_r'];

    covar_type = "diag"
    if covar_type == "diag":
        # Covariance matrices *must* be diagonal to avoid problems with positive-definite, symmetric requirements
        size = sigma_w.shape[0]
        sigma_w = np.multiply( np.identity(size), sigma_w).diagonal()
        sigma_n = np.multiply( np.identity(size), sigma_n).diagonal()
        sigma_r = np.multiply( np.identity(size), sigma_r).diagonal()

    # TODO: set negative values to 0
    model = hmm.GaussianHMM(n_components=3, covariance_type=covar_type, n_iter=100)

    model.means_ = np.array([ mu_w, mu_n, mu_r ])
    model.covars_ = np.array([sigma_w, sigma_n, sigma_r])

    # State order: W, N, R
    start_probs = np.array([ 0.6, 0.4, 0.0 ])
    assert np.sum(start_probs) == 1

    model.startprob_= start_probs
    model.transmat_ = np.load(trans_mat_file).transpose()

    # 3) Predict the labels of the feature matrix
    # model.predict(X) where X: array-like, shape (n_samples, n_features)
    # Returns Label matrix L where L is array of n_samples labels
    L = model.predict(feat_mat)

    return L

def is_positive_definite(covars):
    """
    Test that covariance matrices are positive-definite and symmetric.
    See this requirement here: http://students.mimuw.edu.pl/~pbechler/numpy_doc/reference/generated/numpy.random.multivariate_normal.html
    :params cv: numpy array of covariance matrices.
    :return: is_positive_definite
    """
    for n, cv in enumerate(covars):
        if (not np.allclose(cv, cv.T) or np.any(np.linalg.eigvalsh(cv) <= 0)):
            print 'Covariance Matrix %d is not positive definite' %n
            return False
    return True


def score_model(L, reduced_label_file):
    """Score the model results
    :param L: model results
    :param reduced_label_file: labels
    """
    stages = get_stages_array(reduced_label_file)

    assert len(stages) == len(L)

    score = 0

    # For definitions, see: https://en.wikipedia.org/wiki/Precision_and_recall
    states = ['W', 'N', 'R']
    for sid, state in enumerate(states):

        relevant_elements = 0
        true_positives = 0
        false_positives = 0

        for i in range(0, len(stages)):
            # L[i] represents the items selected by the model
            # stages[i] represents all the relevant items
            if stages[i] == state:
                relevant_elements += 1
            if L[i] == sid and stages[i] == state:
                true_positives += 1
            if L[i] == sid and stages[i] != state:
                false_positives += 1

        print "***** Scoring state %s *****" %state
        print "Precision for state [%s]:%.1f%%" % ( state, 100*true_positives / float(true_positives + false_positives ) )
        print "Recall for state [%s]:%.1f%%" % ( state, 100*true_positives / float(relevant_elements) )
        print "\n"


    print "States labeled: %d - %d" %(np.min(L), np.max(L))

def parse_filename(label_file):
    drive, path_and_file = os.path.splitdrive(label_file)
    path, file = os.path.split(path_and_file)
    session_num = file.split('-')[1].split('.')[0]
    file_no_extension = file.split('.')[0]

    return path, file, session_num, file_no_extension

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Wrong number of arguments."
        print "Usage: python test_model.py <test_data_file.edf> <reduced_label_file.csv>"
        print "Example: python test_model.py ./data/edfs/shhs1-200002.edf ./data/annotations/shhs1-200002-staging-reduced.csv"

        sys.exit()

    test_data_fname = sys.argv[1]
    reduced_label_file = sys.argv[2]

    L = predict_labels(test_data_fname)
    score_model(L, reduced_label_file)