class DONMF_AE(preprocessing):
    def __init__(self, dataset_name, k_neigh, sigma, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, layers, t1=1, t2=1, pre_iter= 500, iter= 500, loss_cal= False):
        super().__init__(dataset_name, k_neigh, sigma)
        self.W, self.D = self.knn_graph()
        self.X = self.X.to('cuda')
        self.XXT = self.X @ self.X.T
        self.pre_iterations = pre_iter
        self.iterations = iter
        self.a1, self.a2, self.a3, self.a4, self.a5, self.a6 = alpha1, alpha2, alpha3, alpha4, alpha5, alpha6
        self.layers = layers
        self.p = layers.numel()
        self.U_s, self.V_s = [0] * self.p, [0] * self.p
        self.loss_cal = loss_cal
        self.t1 = float(t1)
        self.t2 = float(t2)


    # Low-dimensional embedding in i-th layer
    def setup_Z(self, i):
        if i == 0:
            self.Z = self.X
        else:
            self.Z = reduce(torch.matmul, self.V_s[:i]).T @ self.X

    # Shallow AE-like NMF for pre-training
    def auto_encoder_pretrain(self, i):
        # torch.manual_seed(40)
        U = torch.rand(self.Z.shape[0], self.layers[i]).cuda()
        V = torch.rand(self.Z.shape[0], self.layers[i]).cuda()
        # U = 1e-1 * torch.ones((self.Z.shape[0], self.layers[i]), dtype= torch.float, device= 'cuda')
        # V = 1e-1 * torch.ones((self.Z.shape[0], self.layers[i]), dtype= torch.float, device= 'cuda')

        for i in range(self.pre_iterations):
            self.ZZT = self.Z @ self.Z.T
            
            u_u = 2 * self.ZZT @ V 
            u_d = U @ V.T @ self.ZZT @ V + self.ZZT @ U
            U = U * (u_u / torch.maximum(u_d, torch.full_like(u_d, 1e-10)))

            v_u = 2 * self.ZZT @ U
            v_d = self.ZZT @ V @ U.T @ U + self.ZZT @ V
            V = V * (v_u / torch.maximum(v_d, torch.full_like(v_d, 1e-10)))
        return U, V
    
    # Pre-training process
    def pre_training(self):
        for i in range(self.p):
            self.setup_Z(i)
            U, V = self.auto_encoder_pretrain(i)
            self.U_s[i] = U
            self.V_s[i] = V
        
        self.H = reduce(torch.matmul, self.V_s).T @ self.X

    # Defining bipartite graph weight matrix A
    def bipartite_weight(self):
        # self.A = torch.exp(norm(self.X[:, torch.newaxis, :] - self.Ub[:, :, torch.newaxis], dim= 0) / self.landa)
        self.A = 1 / torch.maximum(self.U_tilde[0].T @ self.X, torch.full(size= (self.U_tilde[0].shape[1], self.X.shape[1]), fill_value= 1e-10).cuda()) 
    

    # Defining repulsive weight matrix S
    def repulsive(self):
        self.bipartite_weight()
        column_norms_A = norm(self.A.clone().detach(), dim= 0, keepdim= True)
        normalized_A = self.A.clone().detach() / column_norms_A
        self.S = 1 / torch.maximum(normalized_A.T @ normalized_A, torch.full(size= (self.A.shape[1], self.A.shape[1]), fill_value= 1e-10).cuda())


    def update_U(self, i):
        self.V = reduce(torch.matmul, self.V_s)
        self.U = reduce(torch.matmul, self.U_s)
        self.set_hat(i)
        # self.set_tilde()

        top = self.U_hat.T @ (self.lambda1 ** 2) @ self.X @ self.H.T @ self.U_tilde[i+1].T + self.U_hat.T @ self.X @ self.H.T @ (self.lambda2 ** 2) @ self.U_tilde[i+1].T \
        + self.a5 * self.V_s[i] + self.a4 * self.U_hat.T @ self.U @ self.U_tilde[i+1].T
        bottom = self.U_hat.T @ (self.lambda1 ** 2) @ self.U @ self.H @ self.H.T @ self.U_tilde[i+1].T + self.U_hat.T @ self.XXT @ self.U @ (self.lambda2 **2) @ self.U_tilde[i+1].T \
        + self.a5 * self.V_s[i] @ self.V_s[i].T @ self.U_s[i] + self.a4 * self.U_hat.T @ self.U @ torch.ones(size= (self.U.shape[1], self.U.shape[1]), dtype= torch.float, device= 'cuda') @ self.U_tilde[i+1].T

        self.U_s[i] = self.U_s[i] * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))


    def update_V(self, i):
        self.V = reduce(torch.matmul, self.V_s)
        self.U = reduce(torch.matmul, self.U_s)
        self.set_hat(i, forU= False)
        # self.set_tilde(forU= False)

        top = self.a5 * self.U_s[i] + self.a6 * self.V_hat.T @ self.X @ self.H.T @ self.V_tilde[i+1].T
        bottom = self.a5 * self.U_s[i] @ self.U_s[i].T @ self.V_s[i] + self.a6 * self.V_hat.T @ self.XXT @ self.V @ self.V_tilde[i+1].T

        self.V_s[i] = self.V_s[i] * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))

    def update_H(self):
        self.V = reduce(torch.matmul, self.V_s)
        self.U = reduce(torch.matmul, self.U_s)

        top = self.U.T @ (self.lambda1 ** 2) @ self.X + (self.lambda2 ** 2) @ self.U.T @ self.X + self.a1 * self.H @ self.W + self.a6 * self.V.T @ self.X
        bottom = self.U.T @ (self.lambda1 ** 2) @ self.U @ self.H + (self.lambda2 ** 2) @ self.H + self.a1 * self.H @ self.D + self.a2 * self.H @ self.S + self.a3 * self.A \
        + self.a6 * self.H

        self.H = self.H * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))


    # Diagonal weighting matrices for encoder & decoder
    def enc_dec_weight(self, forEncoder= True):
        if forEncoder:
            residual = self.H - self.U_tilde[0].T @ self.X
            self.lambda2 = torch.diag(norm(residual, dim= 1)) / self.t1

        else:
            residual = self.X - self.U_tilde[0] @ self.H
            self.lambda1 = torch.diag(1 / torch.maximum(norm(residual, dim= 1, keepdim= True), torch.full((residual.shape[0], 1), 1e-10).cuda()).ravel()) / self.t2

    # Set hat(^) based on for U or V from 1 to i-1
    def set_hat(self, i, forU= True):
        if i == 0:
            if forU:
                self.U_hat = torch.eye(int(self.X.shape[0]), device= 'cuda')
            else:
                self.V_hat = torch.eye(int(self.X.shape[0]), device= 'cuda')

        else:
            if forU:
                self.U_hat = self.U_hat @ self.U_s[i-1]
            else:
                self.V_hat = self.V_hat @ self.V_s[i-1]


    # Set tilde(~) based on for U or V from i+1 to l
    def set_tilde(self, forU= True):
        if forU:
            self.U_tilde = [None for _ in range(self.p + 1)]
            self.U_tilde[self.p] = torch.eye(int(self.layers[self.p - 1]), device= 'cuda')
            for i in range(self.p - 1, -1, -1):
                self.U_tilde[i] = self.U_s[i] @ self.U_tilde[i + 1]

        else:
            self.V_tilde = [None for _ in range(self.p + 1)]
            self.V_tilde[self.p] = torch.eye(int(self.layers[self.p - 1]), device= 'cuda')
            for i in range(self.p - 1, -1, -1):
                self.V_tilde[i] = self.V_s[i] @ self.V_tilde[i + 1]

    @staticmethod
    def normalization(P):
        summation = torch.sum(P, dim=0)
        return P / summation


    # Main training process after pre-training
    def training(self):

        self.loss = torch.zeros(self.iterations)
        for iter in range(self.iterations):
            self.set_tilde()
            self.set_tilde(forU= False)
            self.enc_dec_weight()
            self.enc_dec_weight(forEncoder= False)
            self.repulsive()
            for i in range(self.p):
                self.update_U(i)
                self.update_V(i)
                self.update_H()

            if self.loss_cal:
                self.loss[iter] = self.calculate_cost()

        if self.loss_cal:
            plt.plot(self.loss)

        low_dim_embed = self.H
        # low_dim_embed = reduce(torch.matmul, self.V_s).T @ self.X
        normalized_low_dim_embed = low_dim_embed / low_dim_embed.sum(dim=0, keepdim= True)
        kmeans = KMeans(self.labels_num).fit(normalized_low_dim_embed.T.cpu())
        self.pred_labels = kmeans.labels_

        return normalized_mutual_info_score(self.Y.numpy(), self.pred_labels), adjusted_rand_score(self.Y.numpy(), self.pred_labels), self.cluster_acc(self.Y.numpy(), self.pred_labels)

    @staticmethod
    def cluster_acc(real_labels, labels):
        permutation = []
        n_clusters = len(np.unique(real_labels))

        labels = np.unique(labels, return_inverse=True)[1]
        for i in range(n_clusters):
            idx = labels == i
            if np.sum(idx) != 0:
                new_label = scipy.stats.mode(real_labels[idx], keepdims= True)[0][0]
                permutation.append(new_label)
        new_labels = [permutation[label] for label in labels]
        return accuracy_score(real_labels, new_labels)