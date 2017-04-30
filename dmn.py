from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell
import numpy as np

import os
import sys
import threading
import time

import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("load_path", None, "load path of word vector")
flags.DEFINE_string("save_path", None, "save path")
flags.DEFINE_string("restore_path", None, "restore saved parameters")
flags.DEFINE_integer("word_vector_size", 200, "embeding size")
flags.DEFINE_integer("dim", 40, "number of hidden units in input module GRU")
flags.DEFINE_integer("epochs", 50, "number of epochs")
flags.DEFINE_string("answer_module", "feedforward", "answer module type: feedforward or recurrent")
flags.DEFINE_string("mode", "train", "mode: train or test. Test mode required load_state")
flags.DEFINE_string("input_mask_mode", "sentence", "input_mask_mode: word or sentence")
flags.DEFINE_integer("memory_hops", 5, "memory GRU steps")
flags.DEFINE_integer("batch_size", 10)
flags.DEFINE_integer("babi_train_id", 1, "babi train task ID")
flags.DEFINE_integer("babi_test_id", 1, "babi test task ID")
flags.DEFINE_integer("l2", 0, "L2 regularization")
flags.DEFINE_boolean("normalize_attention", False, "flag for enabling softmax on attention vector")
flags.DEFINE_integer("log_every", 1, "print information every x iteration")
flags.DEFINE_integer("save_every", 10, "save state every x epoch")
flags.DEFINE_float("dropout", 0.0, "dropout rate (between 0 and 1)")
flags.DEFINE_float("learning_rate", 0.5, "learning rate")
flags.DEFINE_boolean("batch_norm", False, "batch normalization")


class DynamicMemoryNetwork(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self.dictionary = {}
        self.reverse_dictionary = {}
        self.word2vec = utils.load_glove(self._options.dim)
        self.vocab_size = len(self.dictionary)
        self.load_train_data()
        self.load_test_data()

    def load_train_data(self):
        babi_raw = utils.get_babi_train_raw(self._options.babi_train_id)
        self.train_inputs, self.train_questions, self.train_answers, self.train_fact_count, self.train_input_mask = self._process_input(
            babi_raw)

    def load_test_data(self):
        babi_raw = utils.get_babi_test_raw(self._options.babi_test_id)
        self.test_inputs, self.test_questions, self.test_answers, self.test_fact_count, self.test_input_mask = self._process_input(
            babi_raw)

    def _process_input(self, data_raw):
        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []

        for x in data_raw:
            inp = x["C"].lower().split(' ')
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]

            inp_vector = [utils.process_word(word=w,
                                             word2vec=self.word2vec,
                                             dictionary=self.dictionary,
                                             reverse_dictionary=self.reverse_dictionary,
                                             word_vector_size=self.word_vector_size,
                                             to_return="word2vec") for w in inp]

            q_vector = [utils.process_word(word=w,
                                           word2vec=self.word2vec,
                                           dictionary=self.dictionary,
                                           reverse_dictionary=self.reverse_dictionary,
                                           word_vector_size=self.word_vector_size,
                                           to_return="word2vec") for w in q]

            if (self._options.input_mask_mode == 'word'):
                input_mask = [True] * len(inp)
            elif (self._options.input_mask_mode == 'sentence'):
                input_mask = [True if w == '.' else False for w in inp]
                # input_mask = [index for index, w in enumerate(inp) if w == '.']
            else:
                raise Exception("unknown input_mask_mode")
            fact_count = len(input_mask)

            inputs.append(inp_vector)
            questions.append(q_vector)
            # NOTE: here we assume the answer is one word!
            answers.append(utils.process_word(word=x["A"],
                                              word2vec=self.word2vec,
                                              dictionary=self.dictionary,
                                              reverse_dictionary=self.reverse_dictionary,
                                              word_vector_size=self.word_vector_size,
                                              to_return="index"))
            fact_counts.append(fact_count)
            input_masks.append(input_mask)

        return inputs, questions, answers, fact_counts, input_masks

    def generate_next_batch(self, inputs, questions, answers, fact_count, input_mask):
        inputs = tf.constant(inputs)
        questions = tf.constant(questions)
        answers = tf.constant(answers)
        fact_count = tf.constant(fact_count)
        input_mask = tf.constant(input_mask)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self._options.batch_size

        batch_inputs, batch_questions, batch_answers, batch_fact_count, input_mask = tf.train.shuffle_batch(
            [inputs, questions, answers, fact_count, input_mask],
            batch_size=self._options.batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        padded_inputs = tf.contrib.keras.preprocessing.sequence.pad_sequences(batch_inputs, dtype='float32',
                                                                              paddings='post', value=0)
        padded_questions = tf.contrib.keras.preprocessing.sequence.pad_sequences(batch_questions, dtype='float32',
                                                                                 paddings='post', value=0)
        max_fact_count = tf.contrib.keras.backend.max(batch_fact_count)
        padded_input_mask = tf.contrib.keras.preprocessing.sequence.pad_sequences(batch_questions,
                                                                                  maxlen=max_fact_count, dtype='int32',
                                                                                  paddings='post', value=False)
        return padded_inputs, padded_questions, batch_answers, batch_fact_count, max_fact_count, padded_input_mask

    def generate_next_train_batch(self):
        return self.generate_next_batch(self.train_inputs, self.train_questions, self.train_answers,
                                        self.train_fact_count, self.train_input_mask)

    def generate_next_test_batch(self):
        return self.generate_next_batch(self.test_inputs, self.test_questions, self.test_answers, self.test_fact_count,
                                        self.test_input_mask)

    def make_decoder_batch_input(self, input):
        """ Reshape batch data to be compatible with Seq2Seq RNN decoder.
        :param input: Input 3D tensor that has shape [num_batch, sentence_len, wordvec_dim]
        :return: list of 2D tensor that has shape [num_batch, wordvec_dim]
        """
        input_transposed = tf.transpose(input, [1, 0, 2])  # [L, N, V]
        return tf.unpack(input_transposed)

    def build_graph(self):
        inputs, questions, answers, fact_count, max_fact_count, input_mask = self.generate_next_train_batch()

        gru = rnn_cell.GRUCell(self._options.dim)

        input_states, _ = seq2seq.rnn_decoder(self.make_decoder_batch_input(inputs),
                                              gru.zero_state(self._options.batch_size, tf.float32), gru)

        question_states, _ = seq2seq.rnn_decoder(self.make_decoder_batch_input(questions),
                                                 gru.zero_state(self._options.batch_size, tf.float32), gru)
        question = question_states[-1]

        input_states = tf.transpose(tf.pack(input_states), [1, 0, 2])
        facts = []

        for i in range(len(input_states)):
            filtered = tf.boolean_mask(input_states[i, :, :], input_mask[i, :])  # [?, D]
            padding = tf.zeros(tf.pack([max_fact_count - tf.shape(filtered)[0], self._options.dim]))
            facts.append(tf.concat([filtered, padding], 0))  # [F, D]

        facts = tf.unstack(tf.transpose(tf.pack(facts), [1, 0, 2]), num=max_fact_count)

        episode = tf.zeros([self._options.batch_size, self._options.dim])
        memory = tf.identity(question)

        w1 = tf.Variable(tf.random_normal([self._options.dim, self._options.dim * 7], stddev=0.1))
        b1 = tf.Variable(tf.zeros([self._options.dim, 1]))
        w2 = tf.Variable(tf.random_normal([1, self._options.dim], stddev=0.1))
        b2 = tf.Variable(tf.zeros([1, 1]))

        for _ in range(self._options.memory_hops):
            memory = tf.transpose(memory)
            for c in facts:
                c_t = tf.transpose(c)
                q_t = tf.transpose(question)
                vec = tf.concat(0, [c_t, memory, q_t, c_t * q_t, c_t * memory, (c_t - q_t) ** 2,
                                    (c_t - memory) ** 2])  # (7*d, N)

                l1 = tf.matmul(w1, vec) + b1  # (N, d)
                l1 = tf.nn.tanh(l1)
                l2 = tf.matmul(w2, l1) + b2
                l2 = tf.nn.sigmoid(l2)
                g = tf.transpose(l2)
                episode = g * self.gru(c, episode)[0] + (1 - g) * episode
            memory = gru(episode, memory)[0]

        w_a = tf.Varible(tf.random_normal([self._options.dim, self._options.vocab_size], stddev=0.1))
        logits = tf.matmul(memory, w_a)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answers)
        loss = tf.reduce_mean(cross_entropy)
        self.total_loss = loss + self._options.l2 * tf.add_n(tf.get_collection('l2'))

        predicts = tf.cast(tf.argmax(logits, 1), 'int32')
        corrects = tf.equal(predicts, answers)
        num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        self.global_step = tf.Variable(0, name="global_step")
        self.optimizer = tf.train.AdadeltaOptimizer(self._options.learning_rate).minimize(self.total_loss,
                                                                                          global_step=self.global_step)

    def init(self):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self._session.run(init_op)
        print("Initialized")

    def run(self):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self._session, coord=coord)
        average_loss = 0
        while step < self._options.epochs:
            start_time = time.time()
            _, loss_val, step = self._session.run(
                [self.optimizer, self.total_loss, self.global_step])
            if np.isnan(loss_val):
                print("current loss IS NaN. This should never happen :)")
                sys.exit(1)
            duration = time.time() - start_time
            average_loss += loss_val
            if step % 200 == 0 and step > 0:
                average_loss /= 200
                print('Step: %d Avg_loss: %f (%.3f sec)\r' % (step, average_loss, duration), end="")
                sys.stdout.flush()
                average_loss = 0


def main(_):
    if not FLAGS.load_path or not FLAGS.save_path:
        print("--load_path --save_path must be specified")
        sys.exit(1)

    with tf.Graph().as_default(), tf.Session() as session:
        model = DynamicMemoryNetwork(FLAGS, session)


if __name__ == "__main__":
    tf.app.run()
