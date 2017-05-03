from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib.rnn import core_rnn_cell
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
flags.DEFINE_integer("batch_size", 10, "size of batch")
flags.DEFINE_string("babi_train_id", 1, "babi train task ID")
flags.DEFINE_string("babi_test_id", 1, "babi test task ID")
flags.DEFINE_float("l2", 0, "L2 regularization")
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
        self.word2vec = utils.load_glove(self._options.word_vector_size)
        self.load_train_data()
        self.load_test_data()
        self.vocab_size = len(self.word2vec)
        #print("vocab_size ==> %d" % self.vocab_size)
        self.build_graph()

    def load_train_data(self):
        babi_raw = utils.get_babi_train_raw(self._options.babi_train_id)
        self.train_inputs, self.train_questions, self.train_answers, self.train_fact_count, self.train_input_mask = self._process_input(
            babi_raw)
        print("==> training data loaded")
        # print(self.train_inputs.shape)
        # print(self.train_questions.shape)
        # print(self.train_answers.shape)
        # print(self.train_fact_count)
        # print(self.train_input_mask.shape)

    def load_test_data(self):
        babi_raw = utils.get_babi_test_raw(self._options.babi_test_id)
        self.test_inputs, self.test_questions, self.test_answers, self.test_fact_count, self.test_input_mask = self._process_input(
            babi_raw)
        print("==> test data loaded")

    def _process_input(self, data_raw):
        questions = []
        inputs = []
        answers = []
        # fact_counts = []
        input_masks = []
        max_input_len = 0
        max_question_len = 0
        max_fact_count = 0
        for x in data_raw:
            inp = x["C"].lower().split(' ')
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]

            inp_vector = [utils.process_word(word=w,
                                             word2vec=self.word2vec,
                                             dictionary=self.dictionary,
                                             reverse_dictionary=self.reverse_dictionary,
                                             word_vector_size=self._options.word_vector_size,
                                             to_return="word2vec") for w in inp]

            q_vector = [utils.process_word(word=w,
                                           word2vec=self.word2vec,
                                           dictionary=self.dictionary,
                                           reverse_dictionary=self.reverse_dictionary,
                                           word_vector_size=self._options.word_vector_size,
                                           to_return="word2vec") for w in q]
            if max_input_len < len(inp_vector):
                max_input_len = len(inp_vector)
            if max_question_len < len(q_vector):
                max_question_len = len(q_vector)
            if (self._options.input_mask_mode == 'word'):
                mask_vector = [True] * len(inp)
            elif (self._options.input_mask_mode == 'sentence'):
                mask_vector = [True if w == '.' else False for w in inp]
                # input_mask = [index for index, w in enumerate(inp) if w == '.']
            else:
                raise Exception("unknown input_mask_mode")
            if max_fact_count < len(mask_vector):
                max_fact_count = len(mask_vector)
            # fact_count = len(mask_vector)

            inputs.append(np.vstack(inp_vector).astype('float32'))
            questions.append(np.vstack(q_vector).astype('float32'))
            input_masks.append(mask_vector)
            # NOTE: here we assume the answer is one word!
            answers.append(utils.process_word(word=x["A"],
                                              word2vec=self.word2vec,
                                              dictionary=self.dictionary,
                                              reverse_dictionary=self.reverse_dictionary,
                                              word_vector_size=self._options.word_vector_size,
                                              to_return="index"))
            # fact_counts.append(fact_count)

        inputs = [np.pad(inp, ((0, max_input_len - inp.shape[0]), (0, 0)), 'constant',
                         constant_values=0) for inp in inputs]
        inputs = np.asarray(inputs)
        questions = [np.pad(q, ((0, max_question_len - q.shape[0]), (0, 0)), 'constant', constant_values=0) for q in
                     questions]
        questions = np.asarray(questions)
        answers = np.asarray(answers).astype('int32')
        # fact_counts = np.vstack(fact_counts).astype('int32')
        input_masks = [np.pad(m, (0, max_fact_count - len(m)), 'constant', constant_values=False) for m in input_masks]
        input_masks = np.asarray(input_masks)
        return inputs, questions, answers, max_fact_count, input_masks

    def generate_next_batch(self, inputs, questions, answers, input_mask):
        inputs = tf.constant(inputs, name="inputs_const")
        questions = tf.constant(questions, name="questions_const")
        answers = tf.constant(answers, name="answers_const")
        input_mask = tf.constant(input_mask, name="input_mask_const")

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self._options.batch_size
        inputs, questions, answers, input_mask = tf.train.slice_input_producer([inputs, questions, answers, input_mask],
                                                                               num_epochs=self._options.epochs)
        self.batch_inputs, self.batch_questions, self.batch_answers, self.input_mask = tf.train.shuffle_batch(
            [inputs, questions, answers, input_mask],
            batch_size=self._options.batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return self.batch_inputs, self.batch_questions, self.batch_answers, self.input_mask

    def generate_next_train_batch(self):
        return self.generate_next_batch(self.train_inputs, self.train_questions, self.train_answers,
                                        self.train_input_mask)

    def generate_next_test_batch(self):
        return self.generate_next_batch(self.test_inputs, self.test_questions, self.test_answers, self.test_input_mask)

    def make_decoder_batch_input(self, input):
        """ Reshape batch data to be compatible with Seq2Seq RNN decoder.
        :param input: Input 3D tensor that has shape [num_batch, sentence_len, wordvec_dim]
        :return: list of 2D tensor that has shape [num_batch, wordvec_dim]
        """
        input_transposed = tf.transpose(input, [1, 0, 2])  # [L, N, V]
        # return input_transposed
        return tf.unstack(input_transposed)

    def build_graph(self):
        inputs, questions, answers, input_mask = self.generate_next_train_batch()

        gru = core_rnn_cell.GRUCell(self._options.dim)

        with tf.variable_scope('input_question') as scope:
            input_states, _ = seq2seq.rnn_decoder(self.make_decoder_batch_input(inputs),
                                                  gru.zero_state(self._options.batch_size, tf.float32), gru)
            scope.reuse_variables()
            question_states, _ = seq2seq.rnn_decoder(self.make_decoder_batch_input(questions),
                                                     gru.zero_state(self._options.batch_size, tf.float32), gru)
        question = question_states[-1]

        input_states = tf.transpose(tf.stack(input_states), [1, 0, 2])
        facts = []

        for i in range(input_states.shape[0]):
            filtered = tf.boolean_mask(input_states[i, :, :], input_mask[i, :])  # [?,dim]
            padding = tf.zeros([self.train_fact_count - tf.shape(filtered)[0], self._options.dim])
            facts.append(tf.concat([filtered, padding], 0))  # [max_fact_count,dim]

        facts = tf.unstack(tf.transpose(tf.stack(facts), [1, 0, 2]),
                           num=self.train_fact_count)  # [max_fact_count,batch_size,dim]
        with tf.variable_scope('episodic') as scope:
            episode = tf.zeros_like(facts[0], name="episode")  # [batch_size,dim]
            memory = tf.identity(question, name="memory")  # [batch_size,dim]

            w1 = tf.Variable(tf.random_normal([self._options.dim, self._options.dim * 7], stddev=0.1), name="w1")
            b1 = tf.Variable(tf.zeros([self._options.dim, 1]), name="b1")
            w2 = tf.Variable(tf.random_normal([1, self._options.dim], stddev=0.1), name="w2")
            b2 = tf.Variable(tf.zeros([1, 1]), name="b2")
            episode_gru = core_rnn_cell.GRUCell(self._options.dim)
            for _ in range(self._options.memory_hops):
                mem_t = tf.transpose(memory)
                for c in facts:
                    c_t = tf.transpose(c)  # [dim,batch_size]
                    q_t = tf.transpose(question)  # [dim,batch_size]
                    vec = tf.concat([c_t, mem_t, q_t, c_t * q_t, c_t * mem_t, (c_t - q_t) ** 2,
                                     (c_t - mem_t) ** 2], 0)  # [7*dim,batch_size]

                    l1 = tf.matmul(w1, vec) + b1  # [dim,batch_size]
                    l1 = tf.nn.tanh(l1)
                    l2 = tf.matmul(w2, l1) + b2  # [1,batch_size]
                    l2 = tf.nn.sigmoid(l2)
                    g = tf.transpose(l2)  # [batch_size,1]
                    episode = g * episode_gru(c, episode)[0] + (1 - g) * episode  # [batch_size,dim]
                    scope.reuse_variables()
                memory = episode_gru(episode, memory)[0]
                scope.reuse_variables()

        w_a = tf.Variable(tf.random_normal([self._options.dim, self.vocab_size], stddev=0.1), name="w_a")
        logits = tf.matmul(memory, w_a)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answers)
        loss = tf.reduce_mean(cross_entropy)
        self.total_loss = loss + self._options.l2 * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        predicts = tf.cast(tf.argmax(logits, 1), 'int32')
        corrects = tf.equal(predicts, answers)
        # num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

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
        step = 0
        try:
            while not coord.should_stop():
                start_time = time.time()
                _, loss_val, step, accuracy = self._session.run(
                    [self.optimizer, self.total_loss, self.global_step, self.accuracy])
                if np.isnan(loss_val):
                    print("current loss IS NaN. This should never happen :)")
                    sys.exit(1)
                duration = time.time() - start_time
                average_loss += loss_val
                if step % 200 == 0 and step > 0:
                    average_loss /= 200
                    print('Step: %d Avg_loss: %f Accuracy: %f(%.3f sec) \r' % (step, average_loss, accuracy, duration),
                          end="")
                    sys.stdout.flush()
                    average_loss = 0

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (self._options.epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def main(_):
    if not FLAGS.load_path or not FLAGS.save_path:
        print("--load_path --save_path must be specified")
        sys.exit(1)
    with tf.Graph().as_default(), tf.Session() as session:
        model = DynamicMemoryNetwork(FLAGS, session)
        model.init()
        model.run()


if __name__ == "__main__":
    tf.app.run()
