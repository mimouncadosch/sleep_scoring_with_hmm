import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for line in brown_train:
        words = []
        tags = []
        line = line.strip()
        line = line.split()
        for item in line:
            index = item.rfind('/') #find the last '/'
            word = item[0:index]
            tag = item[(index+1):]
            words.append(word)
            tags.append(tag)
        words.append(STOP_SYMBOL) #add STOP_SYMBOL and START_SYMBOLs
        tags.append(STOP_SYMBOL)
        words.insert(0, START_SYMBOL)
        words.insert(0, START_SYMBOL)
        tags.insert(0, START_SYMBOL)
        tags.insert(0, START_SYMBOL)
        brown_words.append(words)
        brown_tags.append(tags)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bi_count = {}
    tri_count = {}
    for line in brown_tags:
        for i in range(len(line)-2): #traverse each line, count bigrams and trigrams
            bituple = (line[i+1],line[i+2])
            trituple = (line[i],line[i+1],line[i+2])
            if bi_count.has_key(bituple) == False:
                bi_count[bituple] = 1
            else:
                bi_count[bituple] += 1
            if tri_count.has_key(trituple) == False:
                tri_count[trituple] = 1
            else:
                tri_count[trituple] += 1
    
    for item in tri_count.keys(): #compute log-probability of each trigram
        if item[0]==START_SYMBOL and item[1]==START_SYMBOL:
            bicount = len(brown_tags)
        else:
            bi = (item[0], item[1])
            bicount = bi_count[bi]
        q_values[item] = math.log(tri_count[item], 2)- math.log(bicount, 2)
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    word_count = {}
    for line in brown_words: #traverse each line, count the occurence time of each word
        for word in line[2:-1]:
            if word_count.has_key(word) == False:
                word_count[word] = 1
            else:
                word_count[word] += 1

    for item in word_count.keys(): #find the known words
        if word_count[item] >= RARE_WORD_MAX_FREQ:
            known_words.add(item)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for line in brown_words:
        line_rare = []
        for word in line[2:-1]: #skip the START_SYMBOLs and STOP_SYMBOL 
            if (word in known_words): #known word
                line_rare.append(word)
            else: #rare word
                line_rare.append(RARE_SYMBOL)
        line_rare.append(STOP_SYMBOL) #add the symbols
        line_rare.insert(0, START_SYMBOL)
        line_rare.insert(0, START_SYMBOL)
        brown_words_rare.append(line_rare)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    e_count = {}
    tag_count = {}
    for i in range(len(brown_words_rare)):
        for j in range(len(brown_words_rare[i])): #traverse over each line, count the tags and emission pairs
            tag = brown_tags[i][j]
            emi = (brown_words_rare[i][j],tag)
            if tag_count.has_key(tag) == False:
                tag_count[tag] = 1
                taglist.add(tag)
            else:
                tag_count[tag] += 1
            if e_count.has_key(emi) == False:
                e_count[emi] = 1
            else:
                e_count[emi] += 1

    for item in e_count.keys(): #compute emission log-probabilities
        tag = item[1]
        e_values[item] = math.log(e_count[item],2) - math.log(tag_count[tag],2)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    taglist.remove(START_SYMBOL) #we should not assign START_SYMBOL and STOP_SYMBOL tags to the words
    taglist.remove(STOP_SYMBOL)
    taglist = list(taglist)
    tagnum = len(taglist)
    for line in brown_dev_words:
        line_rare = []
        viterbi = [] #len(line) rows, len(taglist) columns. each node is (log_prob,prev_tag)
        states = []
        for word in line: #replace the rare words with RARE_SYMBOL
            if (word in known_words):
                line_rare.append(word)
            else:
                line_rare.append(RARE_SYMBOL)

        #initialization. Compute log-probability of each tag for the first word.
        for i in range(tagnum):
            tag = taglist[i]
            if e_values.has_key((line_rare[0],tag)) == True: #emission log-probability
                emi_prob = e_values[(line_rare[0],tag)]
            else: #unseen emission, set (log_prob,prev_tag) to (1,-1), means this node in viterbi is invalid.
                states.append((1,-1))
                continue
            if q_values.has_key((START_SYMBOL,START_SYMBOL,tag))== True: #transition log-probability
                trans_prob = q_values[(START_SYMBOL,START_SYMBOL,tag)]
            else: #unseen transition
                trans_prob = LOG_PROB_OF_ZERO
            prob = emi_prob + trans_prob
            states.append((prob,START_SYMBOL))
        viterbi.append(states)

        #viterbi
        for i in range(1,len(line_rare)): #compute the log-probability of each tag for each word.
            states = []
            word = line_rare[i]
            for j in range(tagnum):
                tag = taglist[j]
                max_trans_prob = -10000
                max_prev_tag = -1
                if e_values.has_key((line_rare[i],tag)) == True: #emission log-probability
                    emi_prob = e_values[(line_rare[i],tag)]
                else:     #unseen emission, set the node invalid
                    states.append((1,-1))
                    continue
                #calculate the maximum transition probability
                for k in range(tagnum):
                    prev_tag = taglist[k] #previous tag
                    prev_prob = viterbi[i-1][k][0] #log-prob of previous node
                    pp_tag_index = viterbi[i-1][k][1] #previous link of the previous tag
                    if prev_prob == 1: #log_prob is 1, means the previous word with taglist[k] is invalid, skip this tag
                        continue
                    if (pp_tag_index==START_SYMBOL):
                        pp_tag = pp_tag_index
                    else:
                        pp_tag = taglist[pp_tag_index]
                    if q_values.has_key((pp_tag,prev_tag,tag))== True: #probability of the trigram tag
                        trans_prob = prev_prob + q_values[(pp_tag,prev_tag,tag)]
                    else: #unseen trigram
                        trans_prob = prev_prob + LOG_PROB_OF_ZERO
                    if trans_prob > max_trans_prob: #update max transition probability
                        max_trans_prob = trans_prob
                        max_prev_tag = k
                prob = emi_prob + max_trans_prob
                states.append((prob, max_prev_tag))
        
            viterbi.append(states)

        #stop. only consider the transition probability
        tag = STOP_SYMBOL
        max_trans_prob = -10000
        max_prev_tag = -1
        for k in range(tagnum):
            prev_tag = taglist[k]
            prev_prob = viterbi[len(line_rare)-1][k][0]
            if prev_prob == 1: #previous tag invalid, skip this tag
                continue
            pp_tag_index = viterbi[len(line_rare)-1][k][1]
            pp_tag = taglist[pp_tag_index] #previous link of the previous tag
            if q_values.has_key((pp_tag,prev_tag,tag))== True: #probability of the trigram tag
                trans_prob = prev_prob + q_values[(pp_tag,prev_tag,tag)]
            else: #unseen trigram
                trans_prob = prev_prob + LOG_PROB_OF_ZERO
            if trans_prob > max_trans_prob: #update max transition probability
                max_trans_prob = trans_prob
                max_prev_tag = k

        #backtrack
        tags = []
        for i in range(len(line_rare)):
            tags.insert(0,max_prev_tag)
            max_prev_tag = viterbi[len(line_rare)-1-i][max_prev_tag][1]
            
        #generate tagged sentence
        tagged_line = ''
        tagged_line = tagged_line + line[0] + '/' + taglist[tags[0]]
        for i in range(1,len(line)):
            tagged_line = tagged_line + ' ' + line[i] + '/' + taglist[tags[i]]
        tagged_line += '\n'
        tagged.append(tagged_line)
        #print tagged_line


    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    
    #set taggers
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for line in brown_dev_words: #tag each line, generate tagged sentence
        tags = trigram_tagger.tag(line)
        tagged_line = ''
        tagged_line = tagged_line + tags[0][0] + '/' + tags[0][1]
        for i in range(1,len(line)):
            tagged_line = tagged_line + ' ' + tags[i][0] + '/' + tags[i][1]
        tagged_line += '\n'
        #print tagged_line
        tagged.append(tagged_line)
    
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
