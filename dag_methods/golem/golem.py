import numpy as np

from dag_methods.golem.model import Model
from dag_methods.golem.trainer import Trainer


class Golem:
    def __init__(self, lambda_1_ev, lambda_1_nv, lambda_2, equal_variances=True,
                 golem_iter=1e+5, learning_rate=1e-3, seed=1):
        self.lambda_1_ev = lambda_1_ev
        self.lambda_1_nv = lambda_1_nv
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.golem_iter = golem_iter
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X=None, cov_emp=None):
        """Solve the unconstrained optimization problem of GOLEM, which involves
            GolemModel and GolemTrainer.

        Args:
            cov_emp (numpy.ndarray): [d, d] empirical covariance matrix.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            golem_iter (int): Number of iterations for training.
            learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
            seed (int): Random seed. Default: 1.

        Returns:
            numpy.ndarray: [d, d] estimated weighted matrix.

        Hyperparameters:
            (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
            (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
        """
        assert (X is not None) ^ (cov_emp is not None), "Input only one of X and cov_emp"
        if X is not None:
            cov_emp = np.cov(X.T, bias=True)

        # Useful variable
        d = len(cov_emp)

        # Equal noise variances
        model_ev = Model(d, self.lambda_1_ev, self.lambda_2,
                         True, self.seed, B_init=None)
        trainer_ev = Trainer(self.learning_rate)
        B_est_ev = trainer_ev.train(model_ev, cov_emp, self.golem_iter)

        # Assuming equal noise variances
        if self.equal_variances:
            return B_est_ev    # Not thresholded yet
        else:    # Assuming non-equal noise variances
            del model_ev
            del trainer_ev
            # Initialize GOLEM-NV with the solution by GOLEM-EV
            model_nv = Model(d, self.lambda_1_nv, self.lambda_2,
                             False, self.seed, B_est_ev)
            trainer_nv = Trainer(self.learning_rate)
            B_est_nv = trainer_nv.train(model_nv, cov_emp, self.golem_iter)
            return B_est_nv    # Not thresholded yet
