# Standard library imports
import os
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from skfeature.function.similarity_based import SPEC, lap_score
from skfeature.utility.construct_W import construct_W
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import FastICA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.featureselection_class import CLASP

class ExperimentalSuite:
    """
    A unified suite for generating sparse clustered data and running clustering baselines.
    """
    def __init__(
        self,
        k: int = 2,
        n_repeats: int = 3,
        out_dir: str = "./benchmark_results",
    ):
        self.k = k
        self.n_repeats = n_repeats
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # ============================================================
    # DATA GENERATING FUNCTION FOR SIMULATION EXPERIMENTS
    # ============================================================

    def generate_data(
        self,
        n=200,
        p=1000,
        d=20,
        k=2,
        delta=1.5,
        rho=0.5,
        gamma=0.6,
        eps=0.0,
        noise_var=2.0,
        seed=42,
    ):
        """
        Simulates sparse cluster data according to Section 4.1
        """
        if p < 2 * d:
            raise ValueError(
                f"Total features p ({p}) must be at least 2*d ({2*d}) "
                f"to fit signal and distractor blocks."
            )

        rng = np.random.default_rng(seed)

        # 1. Variance shift contributed by cluster means (+/- 0.5 * delta)
        mean_var = 0.25 * (delta ** 2)
        total_signal_var = 1.0 + mean_var

        # 2. Cluster Means
        y = rng.choice(np.arange(k), size=n)
        means = np.zeros((k, p))
        
        if k == 2:
            means[0, :d] = -0.5 * delta
            means[1, :d] = +0.5 * delta
        else:
            means[:, :d] = rng.standard_normal(size=(k, d)) * delta
            means -= means.mean(axis=0)

        # 3. Construct Covariance Matrix
        cov_mtx = np.zeros((p, p))

        # SIGNAL BLOCK (d x d): Conditional Variance = 1.0, Off-Diagonal = rho
        signal_cov = np.full((d, d), rho)
        np.fill_diagonal(signal_cov, 1.0)
        cov_mtx[:d, :d] = signal_cov

        # DISTRACTOR BLOCK (d x d): Perfect Decoy
        # Total Variance = 1 + mean_var, Off-Diagonal Covariance = gamma + mean_var
        distractor_cov = np.full((d, d), gamma + mean_var)
        np.fill_diagonal(distractor_cov, total_signal_var)
        cov_mtx[d:2*d, d:2*d] = distractor_cov

        # OPTIONAL PAIRED CROSS-CORRELATION BLOCK (d x d)
        if eps > 0:
            cross_cov = np.diag(np.full(d, eps * np.sqrt(total_signal_var)))
            cov_mtx[:d, d:2*d] = cross_cov
            cov_mtx[d:2*d, :d] = cross_cov

        # INDEPENDENT NOISE BLOCK (p - 2d): Fixed variance sigma^2 = 2.0
        if p > 2 * d:
            np.fill_diagonal(cov_mtx[2*d:, 2*d:], noise_var)

        # 4. Generate Data
        X = np.zeros((n, p))
        for c in range(k):
            idx = np.where(y == c)[0]
            if len(idx) > 0:
                X[idx] = rng.multivariate_normal(
                    mean=means[c],
                    cov=cov_mtx,
                    size=len(idx),
                    method="eigh"
                )

        true_features = np.arange(d)
        return X, y, true_features

    # ============================================================
    # CLUSTERING HELPERS
    # ============================================================

    def run_kmeans(self, X: np.ndarray, k: int = None, seed: int = 42) -> np.ndarray:
        k_val = self.k if k is None else k
        return KMeans(
            n_clusters=k_val,
            n_init=20,
            random_state=seed,
        ).fit_predict(X)

    def run_spectral(self, X: np.ndarray, k: int = None, seed: int = 42) -> np.ndarray:
        k_val = self.k if k is None else k
        sigma = np.median(pairwise_distances(X))

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        return SpectralClustering(
            n_clusters=k_val,
            affinity="rbf",
            gamma=1.0 / (2 * sigma**2),
            assign_labels="kmeans",
            n_init=10,
            random_state=seed,
        ).fit_predict(X)


    # ============================================================
    # FEATURE SELECTION METHODS
    # ============================================================

    def method_ica(self, X: np.ndarray, d: int, k: int = None, seed: int = 42) -> dict:
        k_val = self.k if k is None else k
        model = FastICA(
            n_components=min(d, 10),
            random_state=seed,
            max_iter=1000,
        )
        model.fit(X)

        importance = np.sum(np.abs(model.mixing_), axis=1)
        selected = np.argsort(importance)[-d:]
        X_sel = X[:, selected]
        labels = self.run_kmeans(X_sel, k=k_val, seed=seed)

        return {
            "selected_features": selected,
            "labels": labels,
        }

    def _laplacian_scores(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        W = kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="connectivity",
            include_self=True,
        ).toarray()

        D = np.diag(W.sum(axis=1))
        L = D - W
        scores = []

        for j in range(X.shape[1]):
            f = X[:, j] - np.mean(X[:, j])
            num = f.T @ L @ f
            den = f.T @ D @ f

            if den <= 1e-12:
                scores.append(np.inf)
            else:
                scores.append(num / den)

        return np.array(scores)

    def method_laplacian(self, X: np.ndarray, d: int, k: int = None, seed: int = 42) -> dict:
        k_val = self.k if k is None else k

        # 1. Compute W explicitly (skfeature throws a KeyError if W is omitted)
        W = construct_W(X)

        # 2. Compute Laplacian scores
        score = lap_score.lap_score(X, W=W)

        # 3. Rank features (smaller score = higher feature importance)
        ranking = lap_score.feature_ranking(score)
        selected = ranking[:d]

        # 4. Cluster on selected features
        X_sel = X[:, selected]
        labels = self.run_spectral(X_sel, k=k_val, seed=seed)

        return {
            "selected_features": selected,
            "labels": labels,
        }

    def _mcfs_scores(self, X: np.ndarray, n_clusters: int = 2, n_neighbors: int = 5) -> np.ndarray:
        W = kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="connectivity",
            include_self=True,
        )
        L = laplacian(W, normed=True)
        eigvals, eigvecs = eigh(L.toarray())
        Y = eigvecs[:, 1 : n_clusters + 1]
        
        return np.abs(X.T @ Y).sum(axis=1)

    def method_mcfs(self, X: np.ndarray, d: int, k: int = None, seed: int = 42) -> dict:
        k_val = self.k if k is None else k
        scores = self._mcfs_scores(X, n_clusters=k_val)
        selected = np.argsort(scores)[-d:]
        X_sel = X[:, selected]
        labels = self.run_spectral(X_sel, k=k_val, seed=seed)

        return {
            "selected_features": selected,
            "labels": labels,
        }
    
    def method_spec(self, X: np.ndarray, d: int, k: int = None, seed: int = 42) -> dict:
        k_val = self.k if k is None else k

        # 1. Build a robust affinity matrix (5-NN graph with heat kernel/cosine)
        kwargs_W = {"neighbor_mode": "knn", "k": 5, "metric": "euclidean"}
        W = construct_W(X, **kwargs_W)

        # 2. Compute SPEC scores with explicit W and style (style=0 uses all non-trivial eigenvalues)
        style = 0
        score = SPEC.spec(X, W=W, style=style)

        # 3. Rank features passing the identical style parameter
        ranking = SPEC.feature_ranking(score, style=style)
        selected = ranking[:d]

        # 4. Cluster test subset on top-d features
        X_sel = X[:, selected]
        labels = self.run_spectral(X_sel, k=k_val, seed=seed)

        return {
            "selected_features": selected,
            "labels": labels,
        }

    def method_cae(self, X: np.ndarray, d: int, k: int = None, seed: int = 42) -> dict:
        class ConcreteAutoencoder(nn.Module):
            def __init__(self, input_dim: int, num_features: int):
                super().__init__()
                self.logits = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_features, input_dim)))
                self.decoder = nn.Sequential(
                    nn.Linear(num_features, max(num_features * 2, input_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(num_features * 2, input_dim // 2), input_dim)
                )

            def forward(self, x, temp, hard=False):
                weights = F.gumbel_softmax(self.logits, tau=temp, hard=hard, dim=-1)
                selected_features = torch.matmul(x, weights.t())
                reconstruction = self.decoder(selected_features)
                return reconstruction

        torch.manual_seed(seed)
        np.random.seed(seed)

        epochs = 200
        start_temp = 10.0
        min_temp = 0.01
        batch_size = min(256, len(X))
        lr = 0.01
        
        temp_decay = (min_temp / start_temp) ** (1 / epochs)

        input_dim = X.shape[1]
        model = ConcreteAutoencoder(input_dim=input_dim, num_features=d)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        tensor_X = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor_X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        temp = start_temp
        
        for epoch in range(epochs):
            for (batch_X,) in dataloader:
                optimizer.zero_grad()
                reconstruction = model(batch_X, temp, hard=False)
                loss = F.mse_loss(reconstruction, batch_X)
                loss.backward()
                optimizer.step()
            
            temp *= temp_decay

        model.eval()
        with torch.no_grad():
            logits = model.logits
            selected = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        unique_selected = np.unique(selected)
        if len(unique_selected) < d:
            probs = F.softmax(logits, dim=-1).max(dim=0).values.detach().cpu().numpy()
            selected = np.argsort(probs)[-d:]
        else:
            selected = unique_selected
            
        selected = selected.tolist()

        labels = None
        if k is not None:
            X_selected = X[:, selected]
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(X_selected).tolist()

        return {
            "selected_features": selected,
            "labels": labels,
        }

    def method_groupfs(self, X: np.ndarray, d: int, k: int = None, seed: int = 42) -> dict:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Extract structural dimensions
        N, num_features = X.shape
        C = k if k is not None else max(2, num_features // 10)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Construct feature-wise affinity and normalized Laplacian
        feat_dist = pairwise_distances(X.T, metric='euclidean')
        feat_sigma = np.median(feat_dist) + 1e-5
        W_feat = np.exp(-(feat_dist ** 2) / (2 * feat_sigma ** 2))
        D_feat = np.diag(np.sum(W_feat, axis=1))
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D_feat)) + 1e-8))
        L_feat = np.eye(num_features) - D_inv_sqrt @ W_feat @ D_inv_sqrt
        L_feat_tensor = torch.tensor(L_feat, dtype=torch.float32)
        
        # Spectral initialization for cluster assignments
        eigvals, eigvecs = np.linalg.eigh(L_feat)
        spectral_emb = eigvecs[:, 1:C+1]
        kmeans = KMeans(n_clusters=C, random_state=seed, n_init=10).fit(spectral_emb)
        
        pi_init = np.zeros((num_features, C))
        for i, label in enumerate(kmeans.labels_):
            pi_init[i, label] = 5.0
            
        class GroupFSModel(nn.Module):
            def __init__(self, num_features, C, pi_init):
                super().__init__()
                self.logits = nn.Parameter(torch.tensor(pi_init, dtype=torch.float32))
                self.mu = nn.Parameter(torch.zeros(C))
                self.Q = nn.Parameter(torch.randn(C, C))
                
            def forward(self, X_batch, tau=1.0, sigma=0.1):
                M = F.gumbel_softmax(self.logits, tau=tau, hard=False)
                
                # Deterministic during evaluation (model.eval())
                if self.training:
                    eps = torch.randn_like(self.mu) * sigma
                    z = torch.sigmoid(self.mu + eps)
                else:
                    z = torch.sigmoid(self.mu)
                    
                z_hat = torch.sum(M * z.unsqueeze(0), dim=1)
                X_tilde = X_batch * z_hat.unsqueeze(0)
                return X_tilde, M, z_hat, z

        model = GroupFSModel(num_features, C, pi_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 100
        lambda_1, lambda_2, beta = 1.0, 0.1, 1.0
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            tau = max(0.1, 1.0 - epoch / epochs)
            
            X_tilde, M, z_hat, z = model(X_tensor, tau=tau)
            
            # Sample-wise Smoothness Loss (Optimized matrix multiplication)
            dist_X = torch.cdist(X_tilde, X_tilde, p=2)
            W_X = torch.exp(-(dist_X ** 2) / (2 * torch.median(dist_X)**2 + 1e-5))
            D_X = torch.diag(torch.sum(W_X, dim=1))
            P_X = torch.linalg.solve(D_X + torch.eye(N), W_X)
            L_s = -(1.0 / (N * num_features)) * torch.sum(X_tilde * (P_X @ X_tilde))
            
            # Feature-wise Smoothness Loss
            F_mat = M @ model.Q
            trace_term = torch.trace(F_mat.T @ L_feat_tensor @ F_mat)
            ortho_penalty = torch.norm(F_mat.T @ F_mat - torch.eye(C), p='fro')**2
            L_f = (1.0 / (num_features * C)) * (trace_term + beta * ortho_penalty)
            
            # Group Sparsity Loss
            prob_z_active = torch.sigmoid(model.mu)
            group_sizes = torch.mean(M, dim=0)
            L_reg = torch.mean(prob_z_active * group_sizes)
            
            loss = L_s + lambda_1 * L_f + lambda_2 * L_reg
            loss.backward()
            optimizer.step()

        # Deterministic Final Feature Selection
        model.eval()
        with torch.no_grad():
            _, M_final, _, z_final = model(X_tensor, tau=0.1)
            
            group_scores = z_final.numpy()
            M_np = M_final.numpy()
            feature_labels = np.argmax(M_np, axis=1)
            
            feature_group_scores = group_scores[feature_labels]
            
            feat_var = np.var(X, axis=0)

        # Hierarchical Sorting: Primary = Group Score, Secondary = Feature Variance
        # np.lexsort sorts by (secondary_key, primary_key) in ascending order
        sorted_indices = np.lexsort((feat_var, feature_group_scores))[::-1]
        
        selected = sorted_indices[:d]

        return {
            "selected_features": np.array(selected, dtype=int),
            "labels": feature_labels,
        }

    def method_clasp(
        self, X: np.ndarray, d: int, k: int = None, seed: int = 42, n_jobs: int = 28
    ) -> dict:
        if CLASP is None:
            raise ImportError("UnsupervisedFeatureSelection module not loaded.")

        k_val = self.k if k is None else k
        np.random.seed(seed)

        ufs = CLASP(
            X=X,
            k=k_val,
            target_pct=d / X.shape[1],
            alpha=0.2,
            gamma=0.15,
            M=400,
            lambda_corr=0.3,
            pct_participants=0.2,
            num_participant_samples=50,
            n_jobs=n_jobs,
        )

        pi_final = ufs.select_features()
        selected = np.argsort(pi_final)[-d:]
        X_sel = X[:, selected]
        labels = self.run_spectral(X_sel, k=k_val, seed=seed)

        return {
            "selected_features": selected,
            "labels": labels,
        }

    def get_methods_map(self) -> dict:
        """
        Returns a mapping from method names to instance functions.
        """
        return {
            "ica": self.method_ica,
            "laplacian": self.method_laplacian,
            "mcfs": self.method_mcfs,
            "spec": self.method_spec,
            "cae": self.method_cae,
            "groupfs": self.method_groupfs,
            "clasp": self.method_clasp,
        }