import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    uni_count = {} #The keys are n-gram tuples, the values are their times of occurence.
    bi_count = {}
    tri_count = {}
    uni_total = 0 #total number of words, including STOP (to compute unigram probabilities)
    for line in training_corpus:
        line = line.strip()
        line = line.split()
        line.append(STOP_SYMBOL)
        uni_total += len(line)
        line.insert(0,START_SYMBOL)
        line.insert(0,START_SYMBOL)
        for i in range(len(line)-2): #traverse each sentence, count n-grams
            unituple = (line[i+2], )
            bituple = (line[i+1],line[i+2])
            trituple = (line[i],line[i+1],line[i+2])
            if uni_count.has_key(unituple) == False:
                uni_count[unituple] = 1
            else:
                uni_count[unituple] += 1
            if bi_count.has_key(bituple) == False:
                bi_count[bituple] = 1
            else:
                bi_count[bituple] += 1
            if tri_count.has_key(trituple) == False:
                tri_count[trituple] = 1
            else:
                tri_count[trituple] += 1

    log_uni_total = math.log(uni_total, 2)
    for item in uni_count.keys(): #compute unigram log-probabilities
        unigram_p[item] = math.log(uni_count[item],2) - log_uni_total
    print len(unigram_p)

    for item in bi_count.keys(): #compute bigram log-probabilities
        uni = (item[0], )
        if item[0] == START_SYMBOL: #bigrams that start with START_SYMBOL
            unicount = len(training_corpus)
        else:
            #print uni
            unicount = uni_count[uni]
        bigram_p[item] = math.log(bi_count[item], 2)- math.log(unicount, 2)
    print len(bigram_p)
    
    for item in tri_count.keys():  #compute trigram log-probabilities
        if item[0]==START_SYMBOL and item[1]==START_SYMBOL: #trigrams start with two START_SYMBOLs
            bicount = len(training_corpus)
        else:
            bi = (item[0], item[1])
            #print bi
            bicount = bi_count[bi]
        trigram_p[item] = math.log(tri_count[item], 2)- math.log(bicount, 2)
    print len(trigram_p)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    if n == 1: #unigram
        for line in corpus:
            line = line.strip()
            line = line.split()
            line.append(STOP_SYMBOL)
            s = 0
            for item in line: #traverse the sentence, compute log-probability
                unituple = (item, )
                s = s + ngram_p[unituple]
            scores.append(s)
            #print s
    elif n == 2: #bigram
        for line in corpus:
            line = line.strip()
            line = line.split()
            line.append(STOP_SYMBOL)
            line.insert(0,START_SYMBOL) #add one START_SYMBOL
            bigram = list(nltk.bigrams(line))
            s = 0
            for item in bigram:
                s = s + ngram_p[item]
            scores.append(s)
            #print s
    else: #trigram
        for line in corpus:
            line = line.strip()
            line = line.split()
            line.append(STOP_SYMBOL)
            line.insert(0,START_SYMBOL) #add two START_SYMBOL
            line.insert(0,START_SYMBOL)
            trigram = list(nltk.trigrams(line))
            s = 0
            for item in trigram:
                s = s + ngram_p[item]
            scores.append(s)
            #print s
    return scores


# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for line in corpus:
        s = 0
        line = line.strip()
        line = line.split()
        line.append(STOP_SYMBOL)
        line.insert(0,START_SYMBOL) #for trigram, add two START_SYMBOLs
        line.insert(0,START_SYMBOL)
        for i in range(len(line)-2):
            trigram = (line[i],line[i+1],line[i+2])
            bigram = (line[i+1],line[i+2])
            unigram = (line[i+2], )
            #unseen n-grams, set log-probability to -1000
            if (unigrams.has_key(unigram) == False) or (bigrams.has_key(bigram) == False) or (trigrams.has_key(trigram) == False):
                s = -1000
                break
            else: #compute log-probability using interpolation
                s += math.log(1,2)-math.log(3,2)+math.log((math.pow(2, unigrams[unigram]) + math.pow(2, bigrams[bigram]) + math.pow(2, trigrams[trigram])), 2)
        scores.append(s)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
