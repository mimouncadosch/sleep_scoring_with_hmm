import sys
from estimate_transition_probs import estimate_transition_probs
from estimate_emission_probs import estimate_emission_probs
from test_model import predict_labels, score_model
from edflib import EdfReader

def train(label_file, reduced_label_file, train_data_file, signal_indices):
    print "Training model"
    estimate_transition_probs(label_file)
    estimate_emission_probs(train_data_file, reduced_label_file, signal_indices, mute=True)

def test(reduced_label_file, test_data_file, model_session_num, signal_indices):
    print "Testing model with data from session %s" %model_session_num
    L = predict_labels(test_data_file, model_session_num, signal_indices, mute=True)
    score_model(L, reduced_label_file)


if __name__ == '__main__':

    signal_indices = [5, 6, 7, 8]

    if sys.argv[1] == 'info':
        sn = sys.argv[2]
        train_data_file = './data/edfs/shhs1-' + sn + '.edf'
        e = EdfReader(train_data_file)
        freqs = e.getSignalFreqs()
        labels = e.getSignalTextLabels()
        for i in range(0, len(freqs)):
            print "%d: %s - %d" %(i, labels[i], freqs[i])

    if sys.argv[1] == 'train':
        # print "Usage:"
        # print "python main.py train <label_file.csv> <reduced_label_file.csv> <train_data_file.edf>"
        sn = sys.argv[2]

        label_file = './data/annotations/shhs1-' + sn + '-staging.csv'
        reduced_label_file = './data/annotations/shhs1-' + sn + '-staging-reduced.csv'
        train_data_file = './data/edfs/shhs1-' + sn + '.edf'
        train(label_file, reduced_label_file, train_data_file, signal_indices)


    elif sys.argv[1] == 'test':
        # print "Usage:"
        # print "python main.py test <reduced_label_file.csv> <test_data_file.edf>"

        # Session number of model to use for testing
        model_session_num = sys.argv[2]

        # Session number of test
        test_session_num = sys.argv[3]

        print "Using model build on data from session %s to test data from session %s" %(model_session_num, test_session_num)

        # Label file used for scoring
        reduced_label_file = './data/annotations/shhs1-' + test_session_num + '-staging-reduced.csv'

        # Test data used for predictions
        test_data_file = './data/edfs/shhs1-' + test_session_num + '.edf'
        test(reduced_label_file, test_data_file, model_session_num, signal_indices)

