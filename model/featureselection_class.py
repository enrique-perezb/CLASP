import os

from .stability_class import StabilityLoss

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import time

class UnsupervisedFeatureSelection:

    def __init__(self, 
                 X, 
                 k,
                 rho,
                 alpha,
                 gamma,
                 N,
                 lambda_corr,
                 pct_participants, 
                 num_participant_samples, 
                 n_jobs
                 ):
        print('Starting UFS')
        self.X = X   # nxp
        self.n, self.p = np.shape(X)[0], np.shape(X)[1]
        self.k = k

        self.pct_participants = pct_participants
        self.num_participant_samples = num_participant_samples

        self.rho = rho
        self.alpha = alpha
        self.gamma = gamma
        self.N = N
        self.lambda_corr = lambda_corr

        self.n_jobs=n_jobs

        corr_mtx = np.corrcoef(self.X, rowvar=False)
        corr_mtx = np.nan_to_num(corr_mtx)
        R = np.abs(corr_mtx)
        np.fill_diagonal(R, 0)
        R[R < 0.3] = 0.0
        self.R = R

        self.pi_hist = []
    
    def select_features(self, max_iter=100, patience=20):
        pi = np.ones(shape=self.p) * self.rho
        top_k = int(self.rho * self.p)
        
        patience_counter = 0
        prev_top_features = None

        for it in range(max_iter):
            # 1. Gumbel-top-k sampling
            gumbel_noise = np.random.gumbel(loc=0, scale=1, size=(self.p, self.N))
            log_pi = np.log(pi)
            corr_boost = self.lambda_corr * (self.R @ pi)
            
            sampled_features = []
            for j in range(self.N):
                scores = log_pi + gumbel_noise[:, j] + corr_boost
                idx = np.argsort(scores)[::-1][:top_k]
                sampled_features.append(idx)
            sampled_features = np.array(sampled_features)
            
            # 2. Evaluate stability loss
            stab = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._eval_loss)(feature_set) for feature_set in sampled_features
            )

            # 3. Select elites and update pi
            stab = np.array(stab)
            n_elite = int(np.floor(self.gamma * self.N))
            elites_idx = np.argpartition(stab, -n_elite)[-n_elite:]

            elite_sets = sampled_features[elites_idx]
            indicator = np.zeros(self.p)
            for s in elite_sets:
                indicator[s] += 1
            indicator /= len(elite_sets)

            pi = (1 - self.alpha) * pi + self.alpha * indicator
            pi = np.clip(pi, 1e-3, 1.0)
            self.pi_hist.append(pi)

            # 4. Stopping Criterion: Check Top-K stability
            current_top_features = set(np.argsort(pi)[::-1][:top_k])
            
            if prev_top_features is not None and current_top_features == prev_top_features:
                patience_counter += 1
            else:
                patience_counter = 0
                prev_top_features = current_top_features

            if patience_counter >= patience:
                print(f"Early stopping triggered at iteration {it}: Top {top_k} features unchanged for {patience} iterations.")
                break

        return pi
    
    def _eval_loss(self, feature_set):
        data = self.X[:, feature_set]

        model = StabilityLoss(
            X=data,
            pct=self.pct_participants,
            num_subsamples=self.num_participant_samples,
            k=self.k
        )

        sil_score = model.get_loss()

        return sil_score

