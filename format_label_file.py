from estimate_transition_probs import parse_filename
from dataio import reduce_label_dimensions
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Wrong number of arguments"
        print "Usage: python format_labelfile.py <label_file.csv>"
        print "Example: python format_labelfile.py ./data/annotations/shhs1-200003-staging.csv"

        sys.exit()

    tot_filename = sys.argv[1]
    path, file, session_num, file_no_extension = parse_filename(sys.argv[1])
    
    print "Formatting label file for session number %s" %session_num

    reduced_label_file = path + '/' + file_no_extension + '-reduced.csv'
    reduce_label_dimensions(tot_filename, reduced_label_file)

    print "Saving labeled stages file to %s" %reduced_label_file
