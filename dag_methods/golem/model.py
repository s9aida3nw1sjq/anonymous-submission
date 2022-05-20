import tensorflow as tf


class Model:
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    def __init__(self, d, lambda_1, lambda_2, equal_variances=True,
                 seed=1, B_init=None):
        """Initialize self.

        Args:
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """
        self.d = d
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init

        self._build()

    def _build(self):
        """Build tensorflow graph."""
        tf.compat.v1.reset_default_graph()

        # Placeholders and variables
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.cov_emp = tf.compat.v1.placeholder(tf.float32, shape=[self.d, self.d])
        self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        if self.B_init is not None:
            self.B = tf.Variable(tf.convert_to_tensor(self.B_init, tf.float32))
        else:
            self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        self.B = self._preprocess(self.B)
        self.var = tf.Variable(0.3, tf.float32)
        self.var = self.var * self.var

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h

        # Optimizer
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.score)

    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (tf.Tensor): [d, d] weighted matrix.

        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        return tf.linalg.set_diag(B, tf.zeros(B.shape[0], dtype=tf.float32))

    def _compute_likelihood(self):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        I = tf.eye(self.d, dtype=tf.float32)
        if self.equal_variances:    # Assuming equal noise variances
            return 0.5 * self.d * tf.math.log(
                tf.linalg.trace(
                    tf.transpose(I - self.B) @ self.cov_emp @ (I - self.B)
                )
            ) - tf.linalg.slogdet(I - self.B)[1]
        else:    # Assuming non-equal noise variances
            return 0.5 * tf.math.reduce_sum(
                tf.math.log(
                    tf.linalg.tensor_diag_part(
                        tf.transpose(I - self.B) @ self.cov_emp @ (I - self.B)
                    )
                )
            ) - tf.linalg.slogdet(I - self.B)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return tf.norm(self.B, ord=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d
