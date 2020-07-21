"""
the function to reproduce paper: GloVe: Global Vectors for Word Representation, https://nlp.stanford.edu/pubs/glove.pdf
"""
from collections import Counter
from itertools import chain
import numpy as np
import os
import time
from multiprocessing import cpu_count, pool, Manager
from threading import Thread, Lock


class GloVe:
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
    def __init__(self, dimension=100,
                 min_count=5,
                 window=15,
                 learning_rate=0.05,
                 x_max=100,
                 alpha=3/4,
                 max_product=1e8,
                 overflow_buffer_size=1e6,
                 verbose=1):
        self.dimension = dimension
        self.min_count = min_count
        self.window = window
        self.learning_rate = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.max_product = max_product
        self.overflow_buffer_size = overflow_buffer_size
        self.verbose = verbose

    def fit_train(self, text, epochs=1, threads=0, verbose_int=1):
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
        # to follow the paper, use two different embeddings of the vocabulary, and emerge them as the final result.
        self.embedding = {word: np.random.standard_normal((self.dimension,)) for word in self.vocab}
        # why self.embedding is dict and other paras are list? self.embedding will directly be a dict for looking up
        # when training is completed.
        self.bias = [0 for word in self.vocab]
        # gradient
        self.grademb = [np.ones(self.dimension,) for word in self.vocab]
        self.gradbias = [1.0 for word in self.vocab]
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
                    pre = max(0, index-self.window)
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
        start = time.time()
        len_pairs = 0
        for pairs in self.get_pairs(n=4000):
            len_pairs += len(pairs)
        if threads == 0:
            threads = cpu_count()
        q = Manager().Queue(1)
        for epoch in range(epochs):
            # if threads == 0:
            #    threads = cpu_count()
            '''p = pool.Pool(threads)
            pool_pairs = []
            num_pairs = int(len_pairs / threads)
            for t in range(threads-1):
               pool_pairs.append((num_pairs*t, num_pairs*(t+1), q))
            pool_pairs.append((num_pairs*(threads-1), num_pairs*threads+int(len_pairs % threads), q))
            loss = p.map(self._wrap_for_glove_v2, pool_pairs)
            p.close()
            p.join()
            # p = pool.Pool(threads)'''
            len_pairs = 0
            loss = 0
            for pairs in self.get_pairs(n=4000):
                len_ = len(pairs)
                len_pairs += len_
                #  to boost training, select part of the elements in pairs
                np.random.shuffle(pairs)
                select_len = int(len_ * 0.7)
                loss += self._glove(pairs[:select_len])
            # parallel operation
            '''loss = 0
            if threads == 0:
                threads = cpu_count()
            num_pairs = int(len_pairs / threads)
            Th = []
            for i in range(threads):
                time.sleep(1)
                t = Pool(self._glove_v2, (num_pairs*i, num_pairs*(i+1)))
                #t.setDaemon(True)
                t.start()
                Th.append(t)
            t = Thread(target=self._glove_v2, args=(num_pairs*(threads-1), num_pairs*threads+int(len_pairs % threads)))
            #t.setDaemon(True)
            t.start()
            Th.append(t)
            time.sleep(10)
            for t in Th:
                t.join()
                #loss += Th[i].get_result()
                #Th[threads-1].join()
                # loss += np.sum(p.map(self._glove, pool_pairs))'''

            if self.verbose:
                if (epoch+1) % verbose_int == 0:
                    print("{}/{} epochs - cost time {:.0f}s - ETA {:.0f}s - loss: {:.4f} ...".format(str(epoch+1).rjust(len(str(epochs))),
                                 epochs, time.time() - start, (time.time() - start) / (epoch+1) * (epochs - epoch - 1), np.sum(loss) / len_pairs))
        if self.verbose:
            print("training complete, cost time {:.0f}.".format(time.time() - start))

    def _glove(self, _pairs, fast_mode=False, queue=None):
        """
        to update the embedding and bias.
        :param _pairs:
        :return:
        """
        if fast_mode:
            try:
                infor = queue.get(block=False)
            except:
                infor = []
            if infor:
                self.embedding = infor[0]
                self.bias = infor[1]
                self.grademb = infor[2]
                self.gradbias = infor[3]
        cost = 0
        for index, (w1, w2) in enumerate(_pairs):
            ind_1 = self.vocab_index[w1]
            ind_2 = self.vocab_index[w2]
            val = self.check(w1, w2)
            diff = np.matmul(self.embedding[w1], self.embedding[w2]) + self.bias[ind_1] + self.bias[ind_2] - np.log(val)
            fdiff_ = diff * self.W(val)
            cost += diff * fdiff_ * 0.5  # the loss function
            # Adaptive gradient updates (from the source code)
            fdiff_ *= self.learning_rate
            # learning rate times gradient for word vectors
            temp1 = fdiff_ * self.embedding[w2]
            temp2 = fdiff_ * self.embedding[w1]
            # lock
            #Lock().locked()
            # adaptive updates
            self.embedding[w1] -= temp1 / (np.sqrt(self.grademb[ind_1]))
            self.embedding[w2] -= temp2 / (np.sqrt(self.grademb[ind_2]))
            self.grademb[ind_1] += temp1 * temp1
            self.grademb[ind_2] += temp2 * temp2
            # updates for bias terms
            self.bias[ind_1] -= fdiff_ / (np.sqrt(self.gradbias[ind_1]))
            self.bias[ind_2] -= fdiff_ / (np.sqrt(self.gradbias[ind_2]))
            fdiff_ *= fdiff_
            self.gradbias[ind_1] += fdiff_
            self.gradbias[ind_2] += fdiff_
        if fast_mode:
            try:
                queue.put([self.embedding, self.bias, self.grademb, self.gradbias], block=False)
            except:
                pass
        return cost

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

    def get_pairs_v2(self, s_ind, e_ind, n=2000):
        """
        get n pairs for training, for parallel training
        :param e_ind: 终止index
        :param s_ind: 开始index，用于定为数据
        :param n:
        :return:
        """
        pairs_1 = self.cooccur.get_pairs()
        pairs_2 = self.buffer.get_pairs()
        count = 0
        pin = 0
        pairs = []
        for w1, w2 in pairs_1:
            pin += 1
            if pin >= s_ind and pin < e_ind:
                count += 1
                pairs.append((w1, w2))
                if count % n == 0:
                    yield pairs
                    pairs = []
        for w1, w2 in pairs_2:
            pin += 1
            if pin >= s_ind and pin < e_ind:
                pairs.append((w1, w2))
                count += 1
                if count % n == 0:
                    yield pairs
                    pairs = []
        if pin >= s_ind and pin < e_ind:
            yield pairs

    def _glove_v2(self, s_ind, e_ind, queue):
        """
        to update the embedding and bias. for parallel training
        :param s_ind:
        :param e_ind:
        :return:
        """
        loss = 0
        for pairs in self.get_pairs_v2(s_ind, e_ind, n=4000):
            #  to boost training, select part of the elements in pairs
            len_ = len(pairs)
            np.random.shuffle(pairs)
            select_len = int(len_ * 0.7)
            loss += self._glove(pairs[:select_len], fast_mode=True, queue=queue)
        return loss

    def _wrap_for_glove_v2(self, args):
        return self._glove_v2(*args)


class Pool(Thread):
    """
    for parallel running
    """
    def __init__(self, func, args=()):
        super(Pool, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


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
    overflow buffer of the co-occurrence matrix. Save buffer in the cache file if buffer is full, label the buffer number
    and load buffer, lookup word pairs if needed. The word pairs will be sorted by their production of frequency rank and
    stored in cache file.
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
    gv = GloVe(max_product=1e8, min_count=5, window=15, learning_rate=0.001)
    gv.fit_train(x_all, epochs=3, verbose_int=1, threads=0)
