# preprocess.py
import numpy as np
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sen_len, w2v_path="./model/w2v_all.model"):
        self.w2v_path = w2v_path
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
        self.embedding = None
        self.embedding_dim = None


    def get_w2v_model(self):
        """
        Load the trained Word2Vec model
        """
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size


    def _add_embedding(self, word):
        """
        Add a special token to the embedding.
        """
        # Randomly generate a vector for the special token
        vector = np.random.randn(1, self.embedding_dim)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = np.vstack((self.embedding_matrix, vector))


    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        
        self.word2idx = self.embedding.wv.vocab
        self.idx2word = self.embedding.wv.index2word
        self.embedding_matrix = self.embedding.wv.syn0
        
        # Add "<PAD>" and "<UNK>" to the embedding
        self._add_embedding("<PAD>")
        self._add_embedding("<UNK>")

        print(f"total words: {self.embedding_matrix.shape[0]}")

        return self.embedding_matrix
    

    def pad_sequence(self, sentence):
        """
        Make the input sentence to the preset length.
        If the sentence is longer than the threshold, we only keep the first sen_len part.
        If the sentence is shorter than the threshold, we pad <PAD> token at the end.

        Inputs:
        - sentence: List. 

        Outputs:
        - sentence: List
        """
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])

        assert len(sentence) == self.sen_len
        
        return sentence
    

    def sentence_word2idx(self, sentence):
        """
        Convert the sentence to corresponding indices

        Inputs:
        - sentence: List of words. 

        Outputs:
        - sentence: (sen_len, ) ndarray.
        """
        sentence_idx = []

        for word in sentence:
            if word in self.word2idx.keys():
                sentence_idx.append(self.word2idx[word].index)
            else:
                sentence_idx.append(self.word2idx["<UNK>"])

        # Pad the sentences to have the same length
        sentence_idx = self.pad_sequence(sentence_idx)

        return np.array(sentence_idx)


    def sentences_word2idx(self, sentences):
        """
        Convert the sentences to corresponding indices

        Inputs:
        - sentence: List of Lists. 

        Outputs:
        - sentence: (N, sen_len) ndarray.
        """
        sentence_list = []
        for i, sen in enumerate(sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_list.append(self.sentence_word2idx(sen))

        print()

        return np.array(sentence_list)