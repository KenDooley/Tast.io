"""
This script runs the first stage of the model generation pipeline:
1) Phrase file generation -- learns what frequent bi-grams appear in the
    corpus and saves them for the dictionary creation step.
2) Dictionary creation -- a dictionary containing all tokens (including
    bigrams) which appear in our corpus more than once.
3) Corpus serialization -- Our documents are stored in sparse matrix bag-of-
    words format.
4) TFIDF training -- learns the TFIDF weights for each token in our dictionary.
5) Latent Semantic Indexing -- dimensionality of our TFIDF weighted corpus is
    reduced using singular value decomposition.

The final output is a dense, lower dimensional representation of our original
sparse, high-dimension TFIDF matrix.  This final LSI matrix is then used
to measure similarities between restaurants along with the Word2Vec document
vectors which are calculated in stage 2.
"""

from gensim import corpora, models
from text_stream import MyText
import logging
import pickle

# Log handling
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def build_phrases():
    """
    This script finds bi-grams in our corpus and stores the results to disk.
    """

    bigram = models.Phrases(MyText())
    bigram.save('../data/bigrams')


def build_dictionary(phrases=None):
    """
    This script generates the dictionary of our corpus and can use an optional
    'phrases' file if bi-grams are also being considered.
    """

    if phrases is None:
        bg_stream = gen_all_sentences()
    else:
        bg_stream = (phrases[sentence] for sentence in gen_all_sentences())

    dictionary = corpora.Dictionary(bg_stream, prune_at=None)
    single_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if
                  docfreq == 1]
    dictionary.filter_tokens(single_ids)
    dictionary.compactify()
    dictionary.save('../data/restaurants.dict')


def serialize_corpus(phrase_file, dict_file, sentences=False):
    """
    This function creates our corpus as a bag of words and saves the result to
    disk.  Also creates lookup dictionaries for fast lookup of metadata for
    specific documents in our corpus.

    INPUT: phrase_file - location of pickled phrase file (string)
           dict_file - location of pickled dictionary file (string)
           sentences - whether to stream corpus as sentences or entire
                       documents (boolean)
    """

    corpus = MyText(phrase_file=phrase_file, dict_file=dict_file,
                    sentences=sentences)
    corpora.MmCorpus.serialize('../data/corpus.mm', corpus)
    file1 = open('../data/name_idx.obj', 'w')
    file2 = open('../data/idx_name.obj', 'w')
    pickle.dump(corpus.name_idx, file1)
    pickle.dump(corpus.idx_name, file2)
    file1.close()
    file2.close()


def initialize_tfidf(corpus):
    """
    Trains the TFIDF weights for our corpus and saves them to disk.

    INPUT: corpus - a gensim corpus object
    """

    tfidf = models.TfidfModel(corpus)
    tfidf.save('../data/model.tfidf')


def initialize_lsi(corpus, tfidf, dictionary, num_factors=400):
    """
    Trains a latent semantic index transformation using our tfidf corpus.
    The result is pickled and stored to disk.

    INPUT: corpus - a gensim corpus object
           tfidf - a gensim tfidf model object
           dictionary - a gensim dictionary object
           num_factors - the number of topics (dimensions) used in the LSI
           decomposition.
    """
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
                          num_topics=num_factors)
    lsi.save('../data/model.lsi')


if __name__ == '__main__':

    build_phrases()
    bigrams = models.Phrases.load('../data/bigrams')
    build_dictionary(bigrams)
    dictionary = corpora.Dictionary.load('../data/restaurants.dict')
    serialize_corpus(phrase_file='../data/bigrams',
                     dict_file='../data/restaurants.dict')
    corpus = corpora.MmCorpus('../data/corpus.mm')
    initialize_tfidf(corpus)
    tfidf_model = models.TfidfModel.load('../data/model.tfidf')
    initialize_lsi(corpus, tfidf_model, dictionary)
