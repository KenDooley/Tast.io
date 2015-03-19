"""
This script runs the second stage of the model generation pipeline where we
create our word2vec model and, from that, calculate our document vectors:
1) Word2Vec model training -- due to memory limitations, I first wrote the
    sentences in the corpus to an intermediate file which could then be
    streamed from into the model.
2) Calculate Doc2Vec matrix -- For each document in the corpus, a document
    vector is calculated from the word2vec vectors of the individual terms
    which appear in that document.  The final matrix in then pickeled
    to disk. This matrix representation was ultimately chosen to be my final
    model.
3) Create final document vectors -- In this step we append the LSI matrix
    calculated in stage one to the Doc2Vec matrix calculated in step 2 above.
"""

from gensim import corpora, models, similarities, matutils
from text_stream import MyText
import numpy as np
import logging
import pickle

# Log handling
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

TFIDF_MODEL = models.TfidfModel.load('../data/model.tfidf')


class SentencesFromFile(object):
    """
    Class for streaming lines of text from a unicode file.

    INPUT: filename - unicode file to be streamed from
    """
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, "r"):
            yield line.decode('utf8').split()


def gen_sentence_file(phrase_file, filename='review_sentences.txt'):
    """
    Function for generating a unicode text file containing all sentences in
    our corpus.  A phrase file can be used in order to consider bigrams
    in the corpus.

    INPUT: filename - file to be written to.
           phrase_file - optional phrase file to use.
    """
    bg_stream = MyText(phrase_file=phrase_file)
    with open(filename, "w") as f:
        for sentence in bg_stream:
            sentence.append(u"\n")
            f.write(u" ".join(sentence).encode('utf8'))


def build_word2vec(filename='review_sentences.txt'):
    """
    Function for training a Word2Vec model.  A generator which streams
    sentences is used as input to the model.

    INPUT: filename - name of file from which to stream sentences.
    """
    bg_stream = SentencesFromFile(filename)
    model = models.Word2Vec(bg_stream, size=300, workers=2)
    model.save('../data/mymodel')


def save_doc_matrix(corpus, w2v_model, dictionary):
    """
    Function for saving the Doc2Vec matrix for our corpus to disk.

    INPUT: corpus - a gensim corpus object
           w2v_model - a trained gensim word2vec model
           dictionary - a gensim dictionary object
    """
    file1 = open('../data/w2v_matrix.obj', 'w')
    pickle.dump(doc_matrix(corpus, w2v_model, dictionary), file1)
    file1.close()


def doc_matrix(corpus, w2v_model, dictionary):
    """
    Function for creating a Doc2Vec matrix for our corpus

    INPUT: corpus - a gensim corpus object
           w2v_model - a trained gensim word2vec model
           dictionary - a gensim dictionary object

    OUTPUT: a numpy array with shape equal to
           (num_documents, num_w2vec_dimensions)
    """
    output = doc2vec(corpus[0], w2v_model, dictionary).T[np.newaxis, :]
    for idx in xrange(1, len(corpus)):
        output = np.append(output, doc2vec(corpus[idx], w2v_model,
                           dictionary).T[np.newaxis, :], axis=0)
    return output


def doc2vec(corpus_doc, w2v_model, dictionary):
    """
    Function for calculating a document vector for a single document in our
    corpus.  This is done by calculating the weighted average word vector
    over all words in the document.  The weightings are determined by each
    word's TFIDF weight.  Words which appear in our corpus < 5 times are
    not considered.

    INPUT: corpus_doc - a single document from a gensim corpus object
           w2v_model - a trained gensim word2vec model
           dictionary - a gensim dictionary object

    OUTPUT: a numpy array of dimension equal to the dimensionality of our
           trained Word2Vec model (300 in this case).
    """
    tot_wgt = 0.0
    for word in TFIDF_MODEL[corpus_doc]:
        try:
            a = w2v_model[dictionary[word[0]]]
            tot_wgt += word[1]
        except:
            logging.warning("%s not in Word2Vec" % word[0])

    output = np.zeros(300)
    for word in TFIDF_MODEL[corpus_doc]:
        try:
            output += w2v_model[dictionary[word[0]]] * (word[1] / tot_wgt)
        except:
            logging.warning("%s not in dictionary" % word[0])
    return output


def save_lsi_w2v_matrix(corpus, tfidf_model, lsi_model,
                        filename='../data/big_matrix_wgt.index'):
    """
    Function which concatenates the LSI matrix created in stage 1 to the
    Doc2Vec matrix created by 'doc_matrix'.  The resulting matrix is pickled
    and saved to disk along with its resulting similarity matrix.

    INPUT: corpus - a gensim corpus object
           tfidf_model - a trained gensim TFIDF model
           lsi_model - a trained gensim LSI model
    """
    lsi = lsi_model[tfidf_model[corpus]]
    numpy_matrix = matutils.corpus2dense(lsi, num_terms=400).T
    with open('../data/w2v_matrix_wgt.obj', 'r') as f:
        mat = pickle.load(f)
    big_matrix = np.append(numpy_matrix, mat, axis=1).T
    big_lsi = matutils.Dense2Corpus(big_matrix)
    index = similarities.MatrixSimilarity(big_lsi)
    file1 = open('../data/big_matrix_wgt.obj', 'w')
    pickle.dump(big_matrix.T, file1)
    file1.close()
    index.save(filename)

if __name__ == '__main__':

    gen_sentence_file(phrase_file='../data/bigrams')
    build_word2vec()
    word2vec = models.Word2Vec.load('../data/mymodel')
    dictionary = corpora.Dictionary.load('../data/restaurants.dict')
    corpus = corpora.MmCorpus('../data/corpus.mm')
    tfidf_model = models.TfidfModel.load('../data/model.tfidf')
    lsi_model = models.LsiModel.load('../data/model.lsi')
    save_doc_matrix(corpus, word2vec, dictionary)
    save_lsi_w2v_matrix(corpus, tfidf_model, lsi_model)
