import utils
import logging
import collections
from itertools import islice
from optparse import OptionParser


def create_model_viterbi_only(sentences):
    model = collections.defaultdict(lambda: collections.defaultdict(int))

    for sentence in sentences:
        for token in sentence:
            model[token.word][token.tag] += 1

    return model


def predict_tags_viterbi_only(sentences, model):

    for sentence in sentences:
        for token in sentence:
            max = 0
            for key, value in model[token.word].items():
                if value > max:
                    max = value
                    winner = key
                token.tag = winner

    return sentences


# Splitting data into tags and tokens


class Token:
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    def __str__(self):
        return "%s/%s" % (self.word, self.tag)


# Creating Python Dictionaries for Sentences and Words

def create_model(sentences):

    tag_a = collections.defaultdict(int)  # tag_a = unigram
    tag_b = collections.defaultdict(
        lambda: collections.defaultdict(int))  # tag_b = bigrams
    tag_words = collections.defaultdict(lambda: collections.defaultdict(int))

    for sentence in sentences:                  # Updating counts of unigrams, tags and words, and tag bigrams
        # Temporarily inserting a sentence-start character so we can count words at beginning of sentence.
        sentence.insert(0, Token('', '<s>'))
        for i, token in enumerate(sentence, 0):
            tag_a[token.tag] += 1                 # Unigrams
            tag_words[token.tag][token.word] += 1   # Tags and words
            if (i+1) < len(sentence):               # Tag bigrams
                tag_b[token.tag][sentence[i+1].tag] += 1
        # Removing our sentence-start token again.
        sentence.pop(0)

    # Defining dictionaries into which to populate probabilities of all delicious Viterbi ingredients
    transition = collections.defaultdict(lambda: collections.defaultdict(int))
    emission = collections.defaultdict(lambda: collections.defaultdict(int))

    # Calculating transition probabilities
    for i, item in enumerate(tag_b.items(), 0):
        org = item[0]
        bi = item[1].items()
        count_1 = tag_a.items()[i][1]
        for n in bi:
            count_2 = n[1]
            prob = (float(count_2)+1) / (float(count_1)+45)
            n = n[0]
            transition[org][n] = prob

    # Calculating emission probabilities
    for i, item in enumerate(tag_words.items(), 0):
        org = item[0]
        bi = item[1].items()
        count_1 = tag_a.items()[i][1]
        for n in bi:
            count_2 = n[1]
            prob = float(count_2) / float(count_1)
            n = n[0]
            emission[org][n] = prob
    # print(emission)
    model = transition, emission  # Passing both back to our model
    return model


def predict_tags(sentences, model):

    tagset = ['NN', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
              'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBN', 'VBP', 'VBG', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', '``', "''", ')', '(', '$', ':', '#']
    final = 0

    for sentence in sentences:

        # Grabbing a list of words in sentence.
        words = [token.word for token in sentence]
        viterbi = {}  # Creating the blank dictionary for this sentence.

        for t in tagset:
            # Creating the matrix with a width of len(sentence)
            viterbi[t] = [0] * len(sentence)

        for i, word in enumerate(words, 0):     # Looping through the sentence
            v = 0

            for t in tagset:
                # Grabbing the correct emission probability for word given t
                em_prob = model[1][t][word]
                if em_prob == 0:                # Taking care of unseen words in testing, part 1
                    em_prob = float(0.0000001)
                marker_t = ''
                baseline = 0

                for tag in tagset:

                    # Grabbing the correct transition probability for current tag "t" given each previous tag "tag"
                    tr_prob = model[0][tag][t]
                    # If it's the first word in the sentence, we calculate differently.
                    if i == 0:
                        tr_prob = model[0]['<s>'][t]
                        consider = em_prob * tr_prob

                    if i >= 1:                          # For all subsequent words
                        prev = viterbi[tag][i-1][0]
                        consider = em_prob * tr_prob * prev

                    if (consider > baseline):
                        baseline = consider
                        marker_t = t

                if baseline > v:
                    v = baseline
                    final = marker_t

                # Update your Viterbi cell here after getting the max!!
                viterbi[t][i] = (baseline, marker_t)

            if i == len(sentence)-1:
                # Save the final tag so we can add it to our taglist.
                sentence[i].tag = final

        ###########################################
        tags = []  # Starting our backpointer method
        m = 0
        tag = ''

        for i in range((len(sentence)-1), -1, -1):  # Iterating backward through the list
            # Appending the last tag in the sentence to our list
            if i == (len(sentence)-1):
                tags.append(sentence[i].tag)
            else:                                   # For all subsequent words, working backwards
                for t in tagset:
                    temp = viterbi[t][i][0]
                    if temp != 0:
                        if viterbi[t][i][0] > m:
                            m = viterbi[t][i][0]
                            tag = viterbi[t][i][1]
                # If we originally had "blank" values - for unknown words.
                if m == 0:
                    for t in tagset:
                        if viterbi[t][i][1] != '':
                            tag = viterbi[t][i][1]

                # Add the final tag value to our reversed list
                tags.append(tag)

        tags.reverse()  # Reversing the list from R-L to L-R
        for i in range(len(sentence)):
            # Zipping the taglist back up to the sentence
            sentence[i].tag = tags[i]

    return sentences


if __name__ == "__main__":
    usage = "usage: %prog [options] GOLD TEST"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--debug", action="store_true",
                      help="turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Please provide required arguments")

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    training_file = args[0]
    training_sentences = utils.read_tokens(training_file)
    test_file = args[1]
    test_sentences = utils.read_tokens(test_file)

    model = create_model(training_sentences)

    model_viterbi_only = create_model_viterbi_only(training_sentences)

    # read sentences again because predict_tags_viterbi_only(...) rewrites the tags
    sents = utils.read_tokens(training_file)
    predictions = predict_tags_viterbi_only(sents, model_viterbi_only)
    accuracy = utils.calc_accuracy(training_sentences, predictions)
    print "Accuracy in training [%s sentences] with Viterbi only : %s " % (
        len(sents), accuracy)

    # read sentences again because predict_tags_viterbi_only(...) rewrites the tags
    sents = utils.read_tokens(test_file)
    predictions = predict_tags_viterbi_only(sents, model_viterbi_only)
    accuracy = utils.calc_accuracy(test_sentences, predictions)
    print "Accuracy in testing [%s sentences] with Viterbi only : %s " % (
        len(sents), accuracy)

    # read sentences again because predict_tags(...) rewrites the tags
    sents = utils.read_tokens(training_file)
    predictions = predict_tags(sents, model)
    accuracy = utils.calc_accuracy(training_sentences, predictions)
    print "Accuracy in training [%s sentences] with HMM and Viterbi : %s " % (
        len(sents), accuracy)

    # read sentences again because predict_tags(...) rewrites the tags
    sents = utils.read_tokens(test_file)
    predictions = predict_tags(sents, model)
    accuracy = utils.calc_accuracy(test_sentences, predictions)
    print "Accuracy in testing [%s sentences] with HMM and Viterbi : %s " % (
        len(sents), accuracy)
