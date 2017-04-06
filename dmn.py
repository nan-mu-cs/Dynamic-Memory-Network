from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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
flags.DEFINE_integer("epoch", 50, "number of epochs")
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
flags.DEFINE_boolean("batch_norm", False, "batch normalization")


class DynamicMemoryNetwork(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self.dictionary = {}
        self.reverse_dictionary = {}
        self.word2vec = utils.load_glove(self._options.dim)
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
                input_mask = range(len(inp))
            elif (self._options.input_mask_mode == 'sentence'):
                input_mask = [index for index, w in enumerate(inp) if w == '.']
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

    def generate_next_train_batch(self):
        inputs = tf.constant(self.train_inputs)
        questions = tf.constant(self.train_questions)
        answers = tf.constant(self.train_answers)
        fact_count = tf.constant(self.train_fact_count)
        input_mask = tf.constant(self.train_input_mask)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self._options.batch_size

        return tf.train.shuffle_batch([inputs, questions, answers, fact_count, input_mask],
                                batch_size=self._options.batch_size, capacity=capacity,
                                min_after_dequeue=min_after_dequeue)

    def generate_next_test_batch(self):
        inputs = tf.constant(self.test_inputs)
        questions = tf.constant(self.test_questions)
        answers = tf.constant(self.test_answers)
        fact_count = tf.constant(self.test_fact_count)
        input_mask = tf.constant(self.test_input_mask)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self._options.batch_size

        return tf.train.shuffle_batch([inputs, questions, answers, fact_count, input_mask],
                                batch_size=self._options.batch_size, capacity=capacity,
                                min_after_dequeue=min_after_dequeue)


def main(_):
    if not FLAGS.load_path or not FLAGS.save_path:
        print("--load_path --save_path must be specified")
        sys.exit(1)

    with tf.Graph().as_default(), tf.Session() as session:
        model = DynamicMemoryNetwork(FLAGS, session)


if __name__ == "__main__":
    tf.app.run()
