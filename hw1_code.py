# IMPORTS
##########
from dataclasses import replace
from wordcloud import WordCloud
from nltk.util import ngrams
import nltk
import time
import numpy as np
import matplotlib.pyplot as plt
from pandas import factorize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tabulate import tabulate
#nltk.download('punkt')
nltk.download('stopwords')
##########

# Globals
# Dict for vector/string storage
global vec2sent
vec2sent = {}
global stop_words
stop_words = set(stopwords.words('english'))

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december"
]

def replace_all_numbers(tokens):
    # Time Benchmark
    timer = time.time()

    # Split tokens
    new_tokens = tokens.split(" ")
    for idx, token in enumerate(new_tokens):
        # replace year
        try:
            number_token = int(token)

            if len(token) == 4:
                new_tokens[idx] = "<year>"
                continue
        except:
            pass

        # replace date days
        try:
            number_token = int(token)

            if new_tokens[idx-1] in MONTHS or new_tokens[idx+1] in MONTHS:
                new_tokens[idx] = "<days>"
                continue
        except:
            pass

        # replace integers
        try:
            number_token = int(token)

            new_tokens[idx] = "<integer>"
            continue
        except:
            pass

        # replace decimals
        try:
            float_token = float(token)

            new_tokens[idx] = "<decimal>"
            continue
        except:
            pass

        # replace other
        if any(map(str.isdigit, token)):
            new_tokens[idx] = "<other>"

    tokens = " ".join(new_tokens)
    time_passed = time.time() - timer
    #print(f"Time taken for replace_all_numbers: {time_passed}\n")
    return tokens

def threshold_three(tok_text):
    # Timer for benchmarking
    timer = time.time() 
    threshold_dict={}
    list_count = []
    count_ = 0
    for i, val in enumerate(tok_text):
        split_val = val.split(" ")
       
        for j in split_val:
            list_count.append(j)
            if j in threshold_dict.keys():
                threshold_dict[j] = threshold_dict[j] + 1
            else:
                threshold_dict[j] = 1
    count_ += len(set(list_count))

    time_passed = time.time() - timer
    print(f"Time passed: {time_passed}")
    return threshold_dict, count_

def filter_threshold(threshold_dict, tok_text):
    global stop_words
    unk_count = 0
    stopword_count = 0
    set_bad_words = []
    vocab = []
    list_count1 = []
    count_1 = 0
    # Timer for benchmarking
    timer = time.time()
    # Loop through keys
    for i in threshold_dict.keys():
        if threshold_dict[i] <= 3:
            set_bad_words.append(i)
    # Cast the list to a set
    set_bad_words = set(set_bad_words) 
    # Loop through tok_text 
    for idx_1, sentence in enumerate(tok_text):
        tokens = sentence.split() # Split sentence into tokens
        # Loop through sentence tokens
        for idx_2, token in enumerate(tokens):
            list_count1.append(token)
            # Check if in bad_words
            if token in set_bad_words:
                # Count Unks
                unk_count += 1
                # if so, replace w/ unk
                tokens[idx_2] = "<unk>"
            
            elif token in stop_words:
                # Incr count of stop words
                stopword_count += 1
            
            else:
                # Append token to vocab set
                vocab.append(token)
        
        # Rejoin into string and replace old sentence
        tok_text[idx_1] = " ".join(tokens) 

        # Old and Slow
        #tok_text = list(map(lambda s: s.replace(i, "<unk>"),tok_text))
        #tok_text = np.char.replace(tok_text, i, "<unk>", count=None)
    time_passed = str(time.time() - timer)
    print(f"Time taken for filter_threshold: {time_passed}\n")
    count_1 += len(set(list_count1))
    vocab = set(vocab)
    # return modified tok_text, vocab, and counts
    return tok_text, vocab, stopword_count, unk_count, count_1

def word_vectorizer_(new_text):
    global vec2sent
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(new_text)
    matrix = vectors.toarray()

    # Loop through vectors and add to global vec2word
    for idx, vec in enumerate(matrix):
        vec2sent[vec.tobytes()] = new_text[idx]

    # Return matrix
    return matrix

def count_split_tokens(x_train, x_valid, x_test):
    x_train_count = 0
    x_valid_count = 0
    x_test_count = 0
    for idx in range(len(x_train)):
        # If index in bounds of x_test 
        if idx < len(x_test):
            # Loop through all 3 sets x_train, x_test, x_valid and Split
            x_test_split = x_test[idx].split()

            # Incr Counts
            x_test_count += len(x_test_split)

        # If index in bounds of x_valid
        if idx < len(x_valid):
            # Split 
            x_valid_split = x_valid[idx].split() 
            # Get Count
            x_valid_count += len(x_valid_split)
        
        # Split
        x_train_split = x_train[idx].split()
        # Incr count
        x_train_count += len(x_train_split)
    
    return x_train_count, x_valid_count, x_test_count 

def show_wordcloud(vocab):
    global stop_words
    word_cloud_stopwords = list(stop_words) + list('unk')
    word_cloud_stopwords = set(word_cloud_stopwords)
    wordcloud =  WordCloud(
        background_color='white',
        stopwords=word_cloud_stopwords,
        max_words=1000,
        max_font_size=24,
        scale=2,
        random_state=1)

    if type(vocab) != str:
        vocab = " ".join(vocab)

    wordcloud = wordcloud.generate(vocab)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def generate_ngram_freq(text, n):
    # Get Ngrams
    text_ngrams = list(ngrams(text, n))
    # Get Top Ngrams
    vec = CountVectorizer(ngram_range=(n,n)).fit(text)
    # Get bag of words
    bag_of_words = vec.transform(text)
    # Sum bag of words
    sum_words = bag_of_words.sum(axis=0)
    # Generate Word Freq from Vocab
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    # Sort by frequency 
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    # Return Freqs of top 10
    return words_freq[:10]

def get_valid_test_vocab_size(x_valid, x_test):
    # Containers to hold tokens
    x_valid_toks = []
    x_test_toks = []
    for idx, x_test_sent in enumerate(x_test): 
        # Add tokens from sent to container
        x_valid_sent = x_valid[idx]
        x_valid_toks += x_valid_sent.split(" ")
        x_test_toks += x_test_sent.split(" ")

    # Convert Lists to Sets
    x_valid_vocab = set(x_valid_toks)
    x_test_vocab = set(x_test_toks)
    # Return
    return x_valid_vocab, x_test_vocab

def get_train_vocab_size(x_train):
    # Container to hold tokens
    x_train_toks = []
    # Loop through x_train 
    for x_train_sent in x_train:
        # Add tokens from sent to container
        x_train_toks += x_train_sent.split(" ")
    # Convert List to Set
    x_train_vocab = set(x_train_toks)
    return x_train_vocab

def tokenize_text(file):
    # TOKENIZATION
    # Reading in all text into a list
    all_text = []
    with open(file, encoding="utf-8") as input_file:
        all_text = input_file.readlines()
        input_file.close()
    # Lower Text
    all_text = [text.lower() for text in all_text]  # converting all text to lowercase
    all_text = "".join(all_text)  # flattening list containing all text, into one string
    # Tokenize Text using NLTK
    tokenized_text = word_tokenize(all_text)
    sent_tokenized_text = sent_tokenize(all_text)
    # Tag Tokens
    sent_tokenized_text = list(map(replace_all_numbers, sent_tokenized_text))
    # Put into Numpy
    tok_text = np.array(sent_tokenized_text)
    # Return
    return tok_text, sent_tokenized_text

def make_labels(text):
    # Make labels
    labels = np.zeros(text.shape)
    return labels

def split_data(data, data_labels, proportions=False): 
    # Split data in testing, training, and validation sets
    x_rem, x_train, y_rem, y_train = train_test_split(data, data_labels, test_size=0.8)
    x_test, x_valid, y_test, y_valid = train_test_split(x_rem, y_rem, test_size = 0.5)
    return x_train, x_test, x_valid, y_train, y_test, y_valid 

def get_num_out_of_vocab_words(train_vocab, test_vocab, valid_vocab):
    num_out_of_vocab_words = len(valid_vocab.difference(train_vocab)) + len(test_vocab.difference(train_vocab))
    return num_out_of_vocab_words

class my_corpus():
    def __init__(self, params):
        super().__init__() 
        
        self.params = params
        print('setting parameters')
    
    def encode_as_ints(self, sequence):
        # Turn input sequence into a list
        sequence = [sequence]
        # Vectorize Input Sequence
        int_represent = word_vectorizer_(sequence) 
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        # Return Vector 
        return(int_represent)
    
    def encode_as_text(self,int_represent):
        # Access Global Lookup
        global vec2sent
        # Get Text
        text = vec2sent[int_represent.tobytes()] 
        print('encode this list', int_represent)
        print('as a text sequence.')
        
        return(text)
    
def main():
    corpus = my_corpus(None)
    
    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ',ints)
    
    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)
    
if __name__ == "__main__":
    main()
        
    
    
              