from curses.ascii import isupper
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import time
import math
import re
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pdb


PUNCTUATION = "-{};:'\,/#$%'*&~"
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

def tokenize_text(dataType:str):
    # TOKENIZATION
    # Reading in all text into a list
    # construct filename
    filename = 'data/wiki' + '.' + dataType + '.txt'
    all_text = []
    with open(filename, encoding="utf-8") as input_file:
        all_text = input_file.readlines()
        input_file.close()
    # Lower Text
    #all_text = [text.lower() for text in all_text]  # converting all text to lowercase
    all_text = "".join(all_text)  # flattening list containing all text, into one string
    print(f"the number of open-parentheses is: {all_text.count('(')}")
    print(f"the number of open-parentheses is: {all_text.count('[')}")
    print(f"the number of open-parentheses is: {all_text.count('{')}")
    # Tokenize Text using NLTK
    #tokenized_text = word_tokenize(all_text)
    sent_tokenized_text = sent_tokenize(all_text)
    # Tag Tokens
    sent_tokenized_text = list(map(replace_all_numbers, sent_tokenized_text))
    sent_tokenized_text = list(map(lambda x: x.replace("\n", "").lstrip(), sent_tokenized_text))
    sent_tokenized_text = " ".join(sent_tokenized_text)

    all_text_split = sent_tokenized_text.split()

    paren_stack = []
    brack_stack = []

    # add end of sequence tokens, tracking parentheses and brackets
    # so that end of sequence tokens are not inserted when
    # parens/brackets are used
    for idx, val in enumerate(all_text_split):
        if val == '(':
            paren_stack.append(val)
        elif val == '[':
            brack_stack.append(val)
        elif val == ')':
            if len(paren_stack) != 0:
                paren_stack.pop(0)
        elif val == ']':
            if len(brack_stack) != 0:
                brack_stack.pop(0)
        elif val == '.' or val == '?' or val == '!' and all_text_split[idx+1][0].isupper():
            if len(paren_stack) == 0 and len(brack_stack) == 0:
                all_text_split[idx] = '</s>'
    
    #word_vectorizer_and_vocab(all_text_split)
    all_text_split = list((map(lambda word: word.lower(), all_text_split)))

    vocab = get_vocab(all_text_split)

    embeddings = random_embeddings(vocab)

    six_word_seq = None

    nLenAllText = len(all_text_split)
    if nLenAllText % 6 == 0:
        all_text_split = [all_text_split[i:i + 6] for i in range(0, len(all_text_split), 6)]
    else:
        nRemainder = nLenAllText % 6
        all_text_split = all_text_split[:nLenAllText - nRemainder]
        the_rest = all_text_split[nLenAllText - nRemainder:]
        the_rest = all_text_split[nLenAllText - nRemainder - 6:nLenAllText - nRemainder]
        all_text_split = [all_text_split[i:i + 6] for i in range(0, len(all_text_split), 6)]
        all_text_split.append(the_rest)

    six_word_seq = all_text_split    

    return six_word_seq, vocab, embeddings


def random_embeddings(vocab):
    """
    vocab: a list of strings, each string is a unique part of the corpus vocabulary
    """
    
    embeddings = np.random.uniform(-0.1, 0.1, (len(vocab), 100))

    return embeddings

def get_vocab(corpus):
    """
    corpus: a list of strings, each string is a token
    """

    return set(corpus)

def truncate(number, decimals=0):
    """
    Taken from: https://kodify.net/python/math/truncate-decimals/ to truncate floats
    
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor    

def word_vectorizer_and_vocab(corpus):
    corpus = [" ".join(corpus)]
    vectorizer = CountVectorizer(analyzer='word', max_features=100)

    vectors = vectorizer.fit_transform(corpus)
    word_embeddings = vectors.toarray()
    vocab = vectorizer.vocabulary_ 
    
    for idx, embedding in enumerate(word_embeddings):
        embedding = np.tanh(embedding)/10
       # pdb.set_trace()
        word_embeddings[idx] = [truncate(feature, 1) for feature in embedding]
    #test_array = [np.tanh(embedding) for embedding in word_embeddings[0:5]]

    return word_embeddings, vocab


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
        
        if token in PUNCTUATION:
            new_tokens[idx] = ""
            
    tokens = " ".join(new_tokens)
    time_passed = time.time() - timer
    #print(f"Time taken for replace_all_numbers: {time_passed}\n")
    return tokens

def make_vocab(sent_list):
    tokens = []
    for sent in sent_list:
        for token in sent:
            tokens.append(token)
    
    return set(tokens)

def make_labels(sequences):
    labels = []
    for sequence in sequences:
        labels.append(sequence[-1])

    return labels

def make_unlabeled(sent_list):
    unlabeled_sents = []
    for sent in sent_list:
        unlabeled_sents.append(sent[0:5])

    return unlabeled_sents


if __name__ == "__main__":
    seqs, vocab, embeddings = tokenize_text('test')




