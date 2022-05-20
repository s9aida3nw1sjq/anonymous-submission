import numpy as np
import tensorflow as tf

from utils.dir import create_dir


class Trainer:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""
    def __init__(self, learning_rate=1e-3):
        """Initialize self.

        Args:
            learning_rate (float): Learning rate of Adam optimizer.
                Default: 1e-3.
        """
        self.learning_rate = learning_rate

    def train(self, model, cov_emp, num_iter):
        """Training and checkpointing.

        Args:
            model (GolemModel object): GolemModel.
            cov_emp (numpy.ndarray): [d, d] empirical covariance matrix.
            num_iter (int): Number of iterations for training.

        Returns:
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True
            )
        )) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(0, int(num_iter) + 1):
                if i == 0:    # Do not train here, only perform evaluation
                    score, likelihood, h, B_est = self.eval_iter(sess, model, cov_emp)
                else:    # Train
                    score, likelihood, h, B_est = self.train_iter(sess, model, cov_emp)

        sess.close()
        return B_est

    def eval_iter(self, sess, model, cov_emp):
        """Evaluation for one iteration. Do not train here.

        Args:
            model (GolemModel object): GolemModel.
            cov_emp (numpy.ndarray): [d, d] empirical covariance matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        score, likelihood, h, B_est \
            = sess.run([model.score, model.likelihood, model.h, model.B],
                        feed_dict={model.cov_emp: cov_emp,
                                   model.lr: self.learning_rate})

        return score, likelihood, h, B_est

    def train_iter(self, sess, model, cov_emp):
        """Training for one iteration.

        Args:
            model (GolemModel object): GolemModel.
            cov_emp (numpy.ndarray): [d, d] empirical covariance matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        _, score, likelihood, h, B_est \
            = sess.run([model.train_op, model.score, model.likelihood, model.h, model.B],
                        feed_dict={model.cov_emp: cov_emp,
                                   model.lr: self.learning_rate})

        return score, likelihood, h, B_est
