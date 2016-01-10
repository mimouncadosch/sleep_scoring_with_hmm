import os
import sys
import csv
import copy
import numpy as np
from dataio import reduce_label_dimensions

"""
Compute transition probabilities matrix A for sleep stages, using MLE.
http://www.ee.columbia.edu/~stanchen/fall12/e6870/slides/lecture4.pdf, slide 35
"""

def compute_counts(reduced_label_file):
    """Compute counts of each possibel transition from state qi to qj
    :param reduced_label_file: name of the file with reduced, labeled sleep stages
    :return A: the transition count matrix
    """

    # Transition Count Matrix in dictionary form
    A = {   'W': {'W':0, 'N': 0, 'R': 0},
            'N': {'W':0, 'N': 0, 'R': 0},
            'R': {'W':0, 'N': 0, 'R': 0}
    }

    # infile = '/Users/mimoun/dsp/data/200002/shhs1-200002-staging-reduced.csv'
    f = open(reduced_label_file, 'r')
    reader = csv.reader(f)

    # Skip header
    reader.next()
    # Initial state
    state_init =  reader.next()[1]
    # Mark count(start, *)
    # A['S'][state_init] += 1

    curr = state_init
    for row in reader:
        nxt = row[1]
        # print "Epoch: %d [curr: %s and next: %s]" %(e, curr, nxt)
        A[curr][nxt] += 1
        curr = nxt

    return A

def compute_transition_probs(A):
    AP = copy.copy(A)

    for curr, nxt_vals in AP.iteritems():
        # Sum of values in row
        tot = np.sum(A[curr].values())
        # Divide each value by total
        for key in nxt_vals:
            nxt_vals[key] = round(nxt_vals[key] / float(tot), 2)

    return AP

# TODO: Save transition probability matrix

def parse_filename(label_file):
    """
    Returns full path, file names, session number, and name of file without the .csv extension
    :param label_file: either full or reduced label file (.csv)
    :return: path, file, session number, file without the extension
    """
    drive, path_and_file = os.path.splitdrive(label_file)
    path, file = os.path.split(path_and_file)
    session_num = file.split('-')[1]
    file_no_extension = file.split('.')[0]

    return path, file, session_num, file_no_extension


def estimate_transition_probs(label_file):
    """Estimates the transition probability matrix given the data in the @label_file.
    Saves the transition matrix to the transition_matrices folder.
    :param label_file: the file with the labels

    """
    path, file, session_num, file_no_extension = parse_filename(label_file)
    print "Computing transition probabilities matrix for session number %s" %session_num

    reduced_label_file = path + '/' + file_no_extension + '-reduced.csv'
    reduce_label_dimensions(label_file, reduced_label_file)

    print "Saving labeled stages file to %s" %reduced_label_file

    A = compute_counts(reduced_label_file)
    AP = compute_transition_probs(A)

    print "Transition Probability Matrix"
    print AP

    # From dictionary to matrix. Order is always W, N, R
    M = np.array([ AP['W']['W'], AP['W']['N'], AP['W']['R'] ])
    M = np.column_stack(( M, [ AP['N']['W'], AP['N']['N'], AP['N']['R'] ] ))
    M = np.column_stack(( M, [ AP['R']['W'], AP['R']['N'], AP['R']['R'] ] ))

    print "Saving transition probability matrix to filename %s" %('./data/transition_matrices/' + session_num + '_transitions.npy')
    np.save('./data/transition_matrices/' + session_num + '_transitions', M)

    return True

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Wrong number of arguments."
        print "Usage: python estimate_transition_probs.py <label_file.csv>"
        print "Example: python estimate_transition_probs.py ./data/annotations/shhs1-200002-staging.csv"

        sys.exit()

    compute_transition_probs(sys.argv[1])

