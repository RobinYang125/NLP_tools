
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import defaultdict
from plotly import tools
import string
import seaborn as sns

def target_count(cnt_srs):
    trace = go.Bar(
        x=cnt_srs.index,
        y=cnt_srs.values,
        marker=dict(
            color=cnt_srs.values,
            colorscale='Picnic',
            reversescale=True
        ),
    )

    layout = go.Layout(
        title='Target Count',
        font=dict(size=18)
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='target_count.html')

def target_distribution(cnt_srs):
    labels = (np.array(cnt_srs.index))
    sizes = (np.array((cnt_srs / cnt_srs.sum()) * 100))

    trace = go.Pie(labels=labels, values=sizes)
    layout = go.Layout(
        title='Target distribution',
        font=dict(size=18),
        width=600,
        height=600,
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='target_distribution.html')


def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0, 16.0),
                   title=None, title_size=40):
    '''
    :param text: pandas.DataFrame
    :param mask:
    :param max_words:
    :param max_font_size:
    :param figure_size:
    :param title:
    :param title_size:
    :param image_color:
    :return:
    '''
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud)
    plt.title(title, fontdict={'size': title_size, 'color': 'black',
                               'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('word_cloud.png')


def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

def n_grams(negative_text, positive_text, n_gram=1):
    freq_dict = defaultdict(int)
    for sent in negative_text:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

    freq_dict = defaultdict(int)
    for sent in positive_text:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

    fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                              subplot_titles=["Frequent words of sincere questions",
                                              "Frequent words of insincere questions"])
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig['layout'].update(height=1200, width=1400, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
    py.plot(fig, filename=f'n_grams_{n_gram}.html')

def meta_data_box(train_df, test_df):
    train_df["num_words"] = train_df["question_text"].apply(lambda x: len(str(x).split()))
    test_df["num_words"] = test_df["question_text"].apply(lambda x: len(str(x).split()))

    ## Number of unique words in the text ##
    train_df["num_unique_words"] = train_df["question_text"].apply(lambda x: len(set(str(x).split())))
    test_df["num_unique_words"] = test_df["question_text"].apply(lambda x: len(set(str(x).split())))

    ## Number of characters in the text ##
    train_df["num_chars"] = train_df["question_text"].apply(lambda x: len(str(x)))
    test_df["num_chars"] = test_df["question_text"].apply(lambda x: len(str(x)))

    ## Number of stopwords in the text ##
    train_df["num_stopwords"] = train_df["question_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    test_df["num_stopwords"] = test_df["question_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    ## Number of punctuations in the text ##
    train_df["num_punctuations"] = train_df['question_text'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    test_df["num_punctuations"] = test_df['question_text'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))

    ## Number of title case words in the text ##
    train_df["num_words_upper"] = train_df["question_text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]))
    test_df["num_words_upper"] = test_df["question_text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]))

    ## Number of title case words in the text ##
    train_df["num_words_title"] = train_df["question_text"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    test_df["num_words_title"] = test_df["question_text"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))

    ## Average length of the words in the text ##
    train_df["mean_word_len"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test_df["mean_word_len"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    ## Truncate some extreme values for better visuals ##
    train_df['num_words'].loc[train_df['num_words'] > 60] = 60  # truncation for better visuals
    train_df['num_punctuations'].loc[train_df['num_punctuations'] > 10] = 10  # truncation for better visuals
    train_df['num_chars'].loc[train_df['num_chars'] > 350] = 350  # truncation for better visuals

    f, axes = plt.subplots(3, 1, figsize=(10, 20))
    sns.boxplot(x='target', y='num_words', data=train_df, ax=axes[0])
    axes[0].set_xlabel('Target', fontsize=12)
    axes[0].set_title("Number of words in each class", fontsize=15)

    sns.boxplot(x='target', y='num_chars', data=train_df, ax=axes[1])
    axes[1].set_xlabel('Target', fontsize=12)
    axes[1].set_title("Number of characters in each class", fontsize=15)

    sns.boxplot(x='target', y='num_punctuations', data=train_df, ax=axes[2])
    axes[2].set_xlabel('Target', fontsize=12)
    # plt.ylabel('Number of punctuations in text', fontsize=12)
    axes[2].set_title("Number of punctuations in each class", fontsize=15)
    # plt.show()
    plt.savefig('meta_data.png')

def main():
    train_df = pd.read_csv('E:\Quora/train.csv')
    test_df = pd.read_csv('E:\Quora/test.csv')

    # cnt_srs = train_df['target'].value_counts()
    # target_count(cnt_srs)
    # target_distribution(cnt_srs)

    # plot_wordcloud(train_df["question_text"], title="Word Cloud of Questions")

    # train1_df = train_df[train_df["target"] == 1]
    # train0_df = train_df[train_df["target"] == 0]
    #
    # n_grams(train0_df['question_text'], train1_df['question_text'], n_gram=3)

    meta_data_box(train_df, test_df)

if __name__ == '__main__':
    main()

    # res = generate_ngrams('There seem to be a variety of words in there. May be it is a good idea to look at the most frequent words in each of the classes separately.', n_gram=3)
    # print(res)