"""
基于Paddle动态图的
the function to reproduce paper: GloVe: Global Vectors for Word Representation, https://nlp.stanford.edu/pubs/glove.pdf
"""
from collections import Counter
from itertools import chain
import numpy as np
import os
import time
from paddle import fluid


class GloVe(fluid.dygraph.Layer):
    """
    dimension: the dimensionality of word embedding.
    min_count: the words with frequency lower than min_count will be neglected.
    window: the window of word co-occurrence, words that co-occur in distance of more than window words will not be counted.
    learning_rate:
    x_max: 1/2 of weight function "W", to weigh the loss of two co-occurred words.
    alpha: 2/2 of weight function "W".
    max_product: Do not easily change this parameter. To limit the product of word rank because the words co-occurrence
    matrix become sparse in the right-bottom, i.e., the word pairs that with very large production of their frequency rank.
    overflow_buffer_size: Don not easily change this parameter. To provide a buffer for the word pairs with production
    exceeding max_product, if the buffer size exceeds overflow_buffer_size, save it as cache file.

    NOTE: the input must be string elements, except type (like int) may cause unpredictable problems.
    """
    def __init__(self, dimension=100, min_count=5, window=15, learning_rate=0.05,
                 x_max=100, alpha=3/4, max_product=1e8, overflow_buffer_size=1e6,
                 use_gpu=True, init_scale=0.1, verbose=1):
        super(GloVe, self).__init__()
        self.dimension = dimension
        self.min_count = min_count
        self.window = window
        self.learning_rate = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.init_scale = init_scale
        self.max_product = max_product
        self.overflow_buffer_size = overflow_buffer_size
        self.verbose = verbose
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        self.built_opt = False  # 标记是否创建了优化器
        self.emb_numpy = None

    def fit_train(self, text, epochs=1, batch_size=4000, verbose_int=10):
        """
        Fit the text data, get the vocabulary of the whole text. Then randomly initialize the embeddings of every word.
        and train text. the fit and train operation are called simultaneous because the fit will get the co-occurrence
        matrix and the matrix only fit the text. when the text changes, the co-occurrence matrix will also changes and
        it will need fit again.
        The embeddings will be stored in a dict formed: {word: embedding, ...}
        :param text: the collects of text, or split words from text. form: [text1, text2, ...] or [[word11, word12, ...],
        [word21, word22, ...], ...]
        :param epochs: training epochs
        :param threads: multiprocessing threads, ==0 will use the max threads of the machine.
        :param verbose_int: the interval of printing information while training.
        :return:
        """
        self._fit(text)
        start = time.time()
        total_step = 0
        total_pairs = 0
        for pairs in self.get_pairs(n=batch_size):
            total_step += 1
            total_pairs += len(pairs)
        for epoch in range(epochs):
            epoch_start = time.time()
            step = 0
            len_pairs = 0
            ave_loss = 0.0
            for pairs in self.get_pairs(n=batch_size):
                len_ = len(pairs)
                step += 1
                np.random.shuffle(pairs)
                loss = self._glove(pairs)
                ave_loss = (loss * len_ + ave_loss * len_pairs) / (len_pairs + len_)
                len_pairs += len_
                if self.verbose:
                    if step % verbose_int == 0:
                        print("{}/{} epochs - {}/{} pairs - ETA {:.0f}s - loss: {:.4f} ...".format(str(epoch+1).rjust(len(str(epochs))),
                                epochs, str(len_pairs).rjust(len(str(total_pairs))), total_pairs, (time.time() - epoch_start) / step * (total_step - step),
                                loss))
            if self.verbose:
                print("{}/{} epochs - cost time {:.0f}s - ETA {:.0f}s - loss: {:.4f} ...".format(str(epoch+1).rjust(len(str(epochs))),
                             epochs, time.time() - start, (time.time() - start) / (epoch+1) * (epochs - epoch - 1), ave_loss))
        if self.verbose:
            print("training complete, cost time {:.0f}s.".format(time.time() - start))

    def forward(self, w_freq, freq, w1, w2):
        """
        core of dygraph logits
        """
        bias_1 = self.bias(w1)
        bias_1 = fluid.layers.reshape(bias_1, shape=(-1, 1))
        bias_2 = self.bias(w2)
        bias_2 = fluid.layers.reshape(bias_2, shape=(-1, 1))
        emb_1 = self.embedding(w1)
        emb_2 = self.embedding(w2)
        mul = fluid.layers.elementwise_mul(emb_1, emb_2)
        mul = fluid.layers.reduce_sum(mul, dim=1)
        diff = mul + bias_1 + bias_2 - fluid.layers.log(freq)
        loss = diff * diff * w_freq
        loss = fluid.layers.reduce_mean(loss)
        return loss

    def _fit(self, text):
        try:
            text = [t.split() for t in text]
            print("The form of input text is [text1, text2, ...].")
        except AttributeError:
            print("The form of input text is [[word11, word12, ...], [word21, word22, ...]].")
        self.words_counter = Counter(chain(*text))
        self.vocab = [word for word, freq in self.words_counter.most_common() if freq > self.min_count]
        if self.verbose:
            print('number of all words: ', len(self.words_counter))
            print('vocabary size: ', len(self.vocab))
        self.vocab_index = {word: index for index, word in enumerate(self.vocab)}
        self.embedding = fluid.Embedding(size=[len(self.vocab_index), self.dimension],
                                         param_attr=fluid.ParamAttr(name='embedding',
                                                                    initializer=fluid.initializer.UniformInitializer(
                                                                        low=-self.init_scale, high=self.init_scale
                                                                    )))
        self.bias = fluid.Embedding(size=[len(self.vocab_index), 1],
                                    param_attr=fluid.ParamAttr(name='bias',
                                                               initializer=fluid.initializer.ConstantInitializer(0.0)))
        # use a sparse form to represent the co-occurrence matrix of words, the high frequency word is the parent of the
        # low frequency
        # word pairs that are d words apart contribute 1/d to the total count. This is one way to account for the fact
        # that very distant word pairs are expected to contain less relevant information about the words’ relationship
        # to one another.
        self.cooccur = CoOccur()
        self.buffer = Buffer(self.overflow_buffer_size)
        total_length = np.sum([freq for freq in self.words_counter.values()])
        if self.verbose:
            print('pre-processing the text, total length (word counts) of the text is ', total_length)
        start = time.time()
        counter = 0
        for num, words in enumerate(text):
            for index, w in enumerate(words):
                if self.words_counter[w] > self.min_count:
                    pre = max(0, index - self.window)
                    length = len(words[pre:index])
                    if length > 0:
                        for i, w_ in enumerate(words[pre:index]):
                            if self.words_counter[w_] > self.min_count:
                                ind_1 = self.vocab_index[w]
                                ind_2 = self.vocab_index[w_]
                                dis = length - i
                                if (ind_1 + 1) * (ind_2 + 1) <= self.max_product:
                                    if ind_1 < ind_2:
                                        self.cooccur.pair(str(w), str(w_), dis)
                                    elif ind_1 == ind_2:
                                        continue
                                    else:
                                        self.cooccur.pair(str(w_), str(w), dis)
                                else:
                                    if ind_1 < ind_2:
                                        self.buffer.pair(str(w), str(w_), dis)
                                    elif ind_1 == ind_2:
                                        continue
                                    else:
                                        self.buffer.pair(str(w_), str(w), dis)
                counter += 1
                if self.verbose:
                    if counter % 100000 == 0:
                        print('{}/{} processed - cost time: {:.0f}s - ETA: {:.0f}s ......'.
                              format(str(counter).rjust(len(str(total_length))), total_length, time.time() - start,
                                     (time.time() - start) / counter * (total_length - counter)))
        self.buffer.update_file()
        if self.verbose:
            print('pre-processing complete. cost time: {:.0f}s'.format(time.time() - start))
        if self.verbose:
            print("fit complete......")
            print("begin training operation......")

    def _glove(self, _pairs):
        """
        to update the embedding and bias.
        :param _pairs:
        :return:
        """
        freq = []
        w1 = []
        w2 = []
        w_freq = []
        for index, (w1_, w2_) in enumerate(_pairs):
            ind_1 = self.vocab_index[w1_]
            ind_2 = self.vocab_index[w2_]
            val = self.check(w1_, w2_)
            w_freq.append(self.W(val))
            freq.append(val)
            w1.append(ind_1)
            w2.append(ind_2)
        freq = np.asarray(freq, dtype='float32')
        w_freq = np.asarray(w_freq, dtype='float32')
        w1 = np.asarray(w1, dtype='int64')
        w2 = np.asarray(w2, dtype='int64')
        freq = fluid.dygraph.to_variable(freq)
        w_freq = fluid.dygraph.to_variable(w_freq)
        w1 = fluid.dygraph.to_variable(w1)
        w2 = fluid.dygraph.to_variable(w2)
        loss = self.forward(w_freq, freq, w1, w2)
        if not self.built_opt:
            self.opt = fluid.optimizer.Adam(learning_rate=self.learning_rate, parameter_list=self.parameters())
            self.built_opt = True
        loss.backward()
        self.opt.minimize(loss)
        return loss.numpy()[0]

    def W(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        else:
            return 1.0

    def check(self, w1, w2):
        """
        search the frequency of w1 and w2
        :param w1:
        :param w2:
        :return:
        """
        ind_1 = self.vocab_index[w1]
        ind_2 = self.vocab_index[w2]
        if ind_1 < ind_2:
            w_1 = w1
            w_2 = w2
        else:
            w_1 = w2
            w_2 = w1
        a = self.cooccur.check(w_1, w_2)
        if a != 0:
            return a
        else:
            return self.buffer.check(w_1, w_2)

    def get_pairs(self, n=2000):
        """
        get n pairs for training
        :param n:
        :return:
        """
        pairs_1 = self.cooccur.get_pairs()
        pairs_2 = self.buffer.get_pairs()
        count = 0
        pairs = []
        for w1, w2 in pairs_1:
            pairs.append((w1, w2))
            count += 1
            if count % n == 0:
                yield pairs
                pairs = []
        for w1, w2 in pairs_2:
            pairs.append((w1, w2))
            count += 1
            if count % n == 0:
                yield pairs
                pairs = []
        yield pairs


class GloVeEval:
    def __init__(self, model):
        self.model = model
        self.emb_numpy = None

    def word_analogy(self, word1, word2, word3, words_list=None, verbose=1):
        """
        word1 is to word2 as word3 is to ? (? in the words_list)
        emb_target = emb_word1 + emb_word2 - emb_word3
        :param verbose: whether or not to show the target(?).
        :param words_list: provide words_list to choose target(?). if is None, the words_list is the vocabulary
        :param word1:
        :param word2:
        :param word3:
        :return:
        """
        target = self.get_embedding(word1) + self.get_embedding(word2) - self.get_embedding(word3)
        target = self.get_similar_word(target, 1, words_list=words_list, verbose=0)
        if verbose:
            print("{} is to {} as {} is to {}".format(word1, word2, word3, target))
        return target

    def get_similarity(self, word1, word2):
        """
        get two the cos similarity of two words
        :param word1:
        :param word2:
        :return:
        """
        emb_1 = self.get_embedding(word1)
        emb_2 = self.get_embedding(word2)
        return np.dot(emb_1, emb_2) / np.sqrt(np.dot(emb_1, emb_1) * np.dot(emb_2, emb_2) + 1e-9)

    def get_similar_word(self, word, k, words_list=None, verbose=1):
        """
        get the top_k most similar words of word in the words_list.
        :param words_list: provide words_list to choose the k most similar words. if is None, the words_list is the vocabulary
        :param verbose: whether (1) or not (0) to print the k words
        :param word: string or embedding
        :param k:
        :return:
        """
        if words_list is None:
            vocab_emb = self.get_vocab_emb()
        else:
            if k > len(words_list):
                raise ValueError(
                    'Not enough words to choose {} most similar words. {} > the length of words_list'.format(k, k))
            vocab_emb = {w: self.get_embedding(w) for w in words_list}
        if isinstance(word, str):
            emb = self.get_embedding(word)
        else:
            emb = word
        word2emb_list = [w for w in vocab_emb.items()]
        vocab_emb = np.array([x[1] for x in word2emb_list])
        cos = np.dot(vocab_emb, emb) / np.sqrt(np.sum(vocab_emb * vocab_emb, axis=1) * np.sum(emb * emb) + 1e-9)
        flat = cos.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        k_words = [word2emb_list[i][0] for i in indices]
        if verbose:
            print('The {} most similar words to {} are(is) {}.'.format(k, word, str(k_words)))
        return k_words

    def get_embedding(self, word):
        """
        get the embedding of word
        :param word:
        :return:
        """
        if self.emb_numpy is None:
            self.emb_numpy = self.model.embedding.parameters()[0].numpy()
        emb = self.emb_numpy
        if word in self.model.vocab_index.keys():
            index = self.model.vocab_index[word]
        else:
            raise KeyError("Can't find word '%s' in dictionary." % word)
        return emb[index]

    def get_vocab_emb(self):
        """
        get the embedding of the vocabulary.
        :return: a dict with words as the keys and embeddings as the values.
        """
        return {w: self.get_embedding(w) for w in self.model.vocab_index.keys()}


class CoOccur:
    """
    store word pairs in memory
    form: {w1: {w2: frequency(w1, w2)}, ...} in which the rank of w1 > rank of w2
    """
    def __init__(self):
        self.cooccur = {}

    def pair(self, w1, w2, dis):
        if w1 in self.cooccur.keys():
            if w2 in self.cooccur[w1].keys():
                self.cooccur[w1][w2] += 1.0 / dis
            else:
                self.cooccur[w1][w2] = 1.0 / dis
        else:
            self.cooccur[w1] = {}
            self.cooccur[w1][w2] = 1.0 / dis

    def check(self, w1, w2):
        if w1 in self.cooccur.keys():
            if w2 in self.cooccur[w1].keys():
                return self.cooccur[w1][w2]
            else:
                return 0
        else:
            return 0

    def get_pairs(self):
        """
        get all word pairs like [(w1, w2), (w3, w4)...]
        :return:
        """
        # return iterator
        return ((w1, w2) for w1 in self.cooccur.keys() for w2 in self.cooccur[w1].keys())


class Buffer:
    """
    overflow buffer of the co-occurrence matrix. Save buffer in the cache file if buffer
    is full, label the buffer number and load buffer, lookup word pairs if needed. The
    word pairs will be sorted by their production of frequency rank and stored in cache file.
    cache_size:
    """
    def __init__(self, size=1e6, cache_path='cache'):
        self.size = size
        self.cooccur = {}
        self.cache_path = cache_path
        self.count = 0
        self.num_saved = 0

    def pair(self, w1, w2, dis):
        if w1 in self.cooccur.keys():
            if w2 in self.cooccur[w1].keys():
                self.cooccur[w1][w2] += 1.0 / dis
            else:
                self.cooccur[w1][w2] = 1.0 / dis
                self.count += 1
        else:
            self.cooccur[w1] = {}
            self.cooccur[w1][w2] = 1.0 / dis
            self.count += 1

        if self.count >= self.size:
            self.update_file()
            pairs = [(w_1, w_2, self.cooccur[w_1][w_2]) for w_1 in self.cooccur.keys() for w_2 in self.cooccur[w_1].keys()]
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
            self.num_saved += 1
            f = open(self.cache_path+'/buffer2bin_' + str(self.num_saved) + '.bin', 'w')
            for pair in pairs:
                f.write(str(pair[0]) + ' ' + str(pair[1]) + ' ' + str(pair[2]) + '\n')
            f.close()
            self.count = 0

    def update_file(self):
        """
        update the saved cache files to avoid the duplicate word pairs.
        :return:
        """
        if self.num_saved > 0:
            for i in range(1, self.num_saved + 1):
                new_f = open(self.cache_path+'/buffer2bin_' + str(i) + 'tem.bin', 'w')
                with open(self.cache_path+'/buffer2bin_' + str(i) + '.bin', 'r') as f:
                    for line in f:
                        w_1, w_2, freq = line.split()
                        freq = float(freq)
                        if w_1 in self.cooccur[w_1]:
                            if w_2 in self.cooccur[w_2]:
                                new_freq = freq + self.cooccur[w_1].pop(w_2)
                                new_f.write(w_1 + ' ' + w_2 + ' ' + str(new_freq) + '\n')
                                if not self.cooccur[w_1]:
                                    self.cooccur.pop(w_1)
                            else:
                                new_f.write(w_1 + ' ' + w_2 + ' ' + str(freq) + '\n')
                        else:
                            new_f.write(w_1 + ' ' + w_2 + ' ' + str(freq) + '\n')
                    f.close()
                new_f.close()
                os.remove(self.cache_path+'/buffer2bin_' + str(i) + '.bin')
                os.rename(self.cache_path+'/buffer2bin_' + str(i) + 'tem.bin', self.cache_path+'/buffer2bin_' + str(i) + '.bin')

    def check(self, w1, w2):
        flag = 0
        while True:
            if w1 in self.cooccur.keys() and flag == 0:
                if w2 in self.cooccur[w1].keys():
                    return self.cooccur[w1][w2]
                else:
                    flag = 1
                    continue
            elif self.num_saved > 0:
                for i in range(1, self.num_saved+1):
                    f = open(self.cache_path+'/buffer2bin_' + str(i) + '.bin', 'r')
                    for line in f:
                        w_1, w_2, freq = line.split()
                        freq = float(freq)
                        if w_1 == w1 and w_2 == w2:
                            f.close()
                            return freq
                    f.close()
                return 0
            else:
                return 0

    def get_pairs(self):
        """
        get all word pairs like [(w1, w2), (w3, w4)...]
        :return:
        """
        if self.num_saved > 0:
            for i in range(1, self.num_saved + 1):
                f = open(self.cache_path+'/buffer2bin_' + str(i) + '.bin', 'r')
                for line in f:
                    w_1, w_2, freq = line.split()
                    yield w_1, w_2
                f.close()
        for w_1 in (i for i in self.cooccur.keys()):
            for w_2 in (j for j in self.cooccur[w_1].keys()):
                yield w_1, w_2


if __name__ == '__main__':
    import chardet

    '''folder_prefix = 'D:/OneDrive/WORK/datasets/'
    x_train = list(open(folder_prefix + "amazon-reviews-train-no-stop.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "amazon-reviews-test-no-stop.txt", 'rb').readlines())
    x_all = []
    x_all = x_all + x_train + x_test
    x_train = list(open(folder_prefix + "r52-train-all-terms.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "r52-test-all-terms.txt", 'rb').readlines())
    x_all = x_all + x_train + x_test
    x_train = list(open(folder_prefix + "r8-train-all-terms.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "r8-test-all-terms.txt", 'rb').readlines())
    x_all = x_all + x_train + x_test
    x_train = list(open(folder_prefix + "20ng-train-all-terms.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "20ng-test-all-terms.txt", 'rb').readlines())
    x_all = x_all + x_train + x_test
    x_train = list(open(folder_prefix + "webkb-train-stemmed.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "webkb-test-stemmed.txt", 'rb').readlines())
    x_all = x_all + x_train + x_test'''
    x_all = list(open("text8", 'rb').readlines())
    x_all = [x_all[0][:3000000]]
    le = len(x_all)
    for i in range(le):
        encode_type = chardet.detect(x_all[i])
        x_all[i] = x_all[i].decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量
    #x_all = [s.split()[1:] for s in x_all]
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        gv = GloVe(max_product=1e8, min_count=5, window=15, learning_rate=0.001)
        gv.fit_train(x_all, epochs=1, batch_size=4000, verbose_int=10)
