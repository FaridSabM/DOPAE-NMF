import torch
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from functools import reduce
from torch.linalg import norm
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd