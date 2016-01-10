import os, sys
import edflib
import numpy as np
from dataio import load_signals, get_stages_array
from epoch import get_epoch_data, number_of_epochs
from features import get_features

def predict_labels(test_data_file, model_session_num, mute):

    # 1) Get data
    e = edflib.EdfReader(test_data_file)
    signal_indices = [5, 7, 8]
    # Signal frequencies
    eogl_f = 50; eeg1_f = 125; resp_f = 10
    eogl, eeg1, resp = load_signals(e, signal_indices)
    num_epochs = number_of_epochs(eeg1, eeg1_f)

    first = True
    for ei in xrange(0, num_epochs):
        # Retrieve data in ei-th epoch
        e1      = get_epoch_data(eeg1, eeg1_f, ei, 1, mute)     # EEG 1
        r       = get_epoch_data(resp, resp_f, ei, 1, mute)     # Respiration
        eol     = get_epoch_data(eogl, eogl_f, ei, 1, mute)     # EOG L
        # Extract the features from the data

        e1_features     = get_features(e1, eeg1_f)
        r_features      = get_features(r, resp_f)
        eogl_features   = get_features(eol, eogl_f)

        # Features is features vector composed of features of many signals stacked together
        features = np.hstack((e1_features, r_features, eogl_features))

        if first is True:
            feat_mat = features
            first = False
        elif first is not True:
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
    start_probs = np.array([0.6, 0.4, 0.0 ])
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

    stages = get_stages_array(reduced_label_file)

    assert len(stages) == len(L)

    score = 0

    for i in xrange(0, len(stages)):
        if L[i] == 0 and stages[i] == 'W':
            score += 1
        elif L[i] == 1 and stages[i] == 'N':
            score += 1
        elif L[i] == 2 and stages[i] == 'R':
            score += 1

    print "Score: %d / %d = %f %%" %(score, len(stages), round(100*float(score)/len(stages),2))

    print np.min(L), np.max(L)

def parse_filename(label_file):
    drive, path_and_file = os.path.splitdrive(label_file)
    path, file = os.path.split(path_and_file)
    session_num = file.split('-')[1].split('.')[0]
    file_no_extension = file.split('.')[0]

    return path, file, session_num, file_no_extension

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Wrong number of arguments."
        print "Usage: python predict_sequence.py <test_data_file.edf> <reduced_label_file.csv>"
        print "Example: python predict_sequence.py ./data/edfs/shhs1-200002.edf ./data/annotations/shhs1-200002-staging-reduced.csv"

        sys.exit()

    test_data_fname = sys.argv[1]
    reduced_label_file = sys.argv[2]

    L = predict_labels(test_data_fname)
    score_model(L, reduced_label_file)

