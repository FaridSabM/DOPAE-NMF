class preprocessing:
    def __init__(self, dataset_name, k_neigh, sigma):
        # data = loadmat(f'C:\\Users\\Asus\\Desktop\\projects\\Deep_Oblique_AE\\Datasets\\{dataset_name}.mat')
        data = loadmat(f'C:\\Users\\Asus\\Desktop\\projects\\Deep_Oblique_AE\\Results\\Noise2\\{dataset_name}.mat')
        self.X = torch.tensor(data['X'], dtype= torch.float).T  # Data matrix
        if self.X.max() > 1:
            self.normalize()
        self.Y = torch.tensor(data['Y']).ravel() - 1 if data['Y'].min() > 0 else torch.tensor(data['Y']).ravel() # Real labels
        self.labels_num = self.Y.unique().numel()
        self.k_neigh = k_neigh
        self.band_width = sigma

    def normalize(self):
        scaler = MinMaxScaler()
        scaler.fit(self.X.T)
        self.X = torch.tensor(scaler.transform(self.X.T).T, dtype= torch.float)
    
    def knn_graph(self):
        d, n = self.X.shape
        index = faiss.IndexFlatL2(d)
        index.add(self.X.T)

        distances, indices = index.search(self.X.T, self.k_neigh + 1)   # Note that this distance is euclidean distance to the power of two
        distances, indices = np.exp(-distances[:, 1:] / self.band_width), indices[:, 1:]
        W = np.zeros((n, n))
        W[np.arange(n)[:, np.newaxis], indices] = distances
        W = np.add(W, W.T) / 2
        D = np.diag(W.sum(axis= 1))
        
        return torch.tensor(W, dtype= torch.float, device= 'cuda'), torch.tensor(D, dtype= torch.float, device= 'cuda')
