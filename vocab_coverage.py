
import operator
import pandas as pd
import pickle as pkl

def build_vocab(texts):
    '''
    :param texts: pandas.DataFrame
    :return: dict{word: count}
    '''
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sent in sentences:
        for word in sent:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embedding_words):
    '''
    :param vocab: dict
    :param embedding_words: dict{word: index}
    :return: dict
    '''
    known_words = {}
    unknown_words = {}
    num_known_words = 0
    num_unknown_words = 0
    for word, count in vocab.items():
        if word in embedding_words:
            known_words[word] = embedding_words[word]
            num_known_words += count
        else:
            unknown_words[word] = count
            num_unknown_words += count
    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(
        num_known_words / (num_known_words + num_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1), reverse=True)
    return unknown_words

def add_lower(embedding, vocab):
    count = 0
    for word in vocab.keys():
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(' ')])
    return text

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')

    # Other special characters that I have to deal with in last
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

    for s in specials:
        text = text.replace(s, specials[s])

    return text


def main():
    train = pd.read_csv('E:\Quora/train.csv').drop('target', axis=1)
    test = pd.read_csv('E:\Quora/test.csv')
    df = pd.concat([train, test])

    with open('C:/Users/v-hayang\Downloads\glove.42B.300d/word2index.pkl', 'rb') as fin:
        word2index = pkl.load(fin)

    # original
    print('original')
    vocab = build_vocab(df['question_text'])
    oov_glove = check_coverage(vocab, word2index)
    print(oov_glove[:10])

    # lower
    print('all lower')
    df['lowered_question'] = df['question_text'].apply(lambda x: x.lower())
    vocab_lower = build_vocab(df['lowered_question'])
    oov_glove = check_coverage(vocab_lower, word2index)
    print(oov_glove[:10])

    # add lower
    print('add lower')
    add_lower(word2index, vocab)
    oov_glove = check_coverage(vocab_lower, word2index)
    print(oov_glove[:10])

    # contractions
    print('contractions')
    contraction_mapping = \
        {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
           "could've": "could have", "couldn't": "could not", "didn't": "did not",
           "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
           "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
           "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
           "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
           "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
           "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
           "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
           "mayn't": "may not", "might've": "might have", "mightn't": "might not",
           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
           "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
           "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
           "she'll've": "she will have", "she's": "she is", "should've": "should have",
           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
           "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
           "that's": "that is", "there'd": "there would", "there'd've": "there would have",
           "there's": "there is", "here's": "here is", "they'd": "they would",
           "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
           "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
           "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
           "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
           "what'll": "what will", "what'll've": "what will have", "what're": "what are",
           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
           "where'd": "where did", "where's": "where is", "where've": "where have",
           "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
           "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
           "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
           "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    df['treated_question'] = df['lowered_question'].apply(
        lambda x: clean_contractions(x, contraction_mapping))
    vocab = build_vocab(df['treated_question'])
    oov_glove = check_coverage(vocab, word2index)
    print(oov_glove[:10])

    # punct
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }

    df['treated_question'] = df['treated_question'].apply(
        lambda x: clean_special_chars(x, punct, punct_mapping))
    vocab = build_vocab(df['treated_question'])
    oov_glove = check_coverage(vocab, word2index)
    print(oov_glove[:10])


if __name__ == '__main__':
    main()