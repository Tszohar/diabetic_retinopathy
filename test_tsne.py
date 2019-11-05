import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import parameters
from data import BDDataset
from resnet import ResNet
import numpy as np

batch_size = 256
model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191104 (14:22:59.927118)_MULTI_CLASS_256_Binary2Multi/model_epoch_98.pth"
colors = ['red', 'green', 'blue', 'purple', 'black', 'cyan']


train_dataset = BDDataset(csv_file=parameters.train_csv, data_dir=parameters.data_dir, transform=None)
train2015_dataset = BDDataset(csv_file=parameters.train2015_csv, data_dir=parameters.data2015_dir, transform=None)
dataset_combined = train_dataset + train2015_dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

net = ResNet()
model = torch.load(model_path)
net.load_state_dict(model)
net.to(parameters.device)
net.eval()

for i_batch, sample_batched in enumerate(train_dataloader):
    outputs = net(sample_batched['image'].to(parameters.device)).detach()
    params_vec = net.vectorize(sample_batched['image']).detach()
    if i_batch % 10 == 0:
        print("batch {}/{}".format(i_batch, len(train_dataloader)))
    if (i_batch == 0):
        params = np.array(params_vec)
        labels = np.array(sample_batched['diagnosis'])
    else:
        labels = np.concatenate((labels, np.array(sample_batched['diagnosis'])), axis=0)
        params = np.concatenate((params, params_vec), axis=0)
pca = PCA(n_components=50, svd_solver='full')
reduced_features = pca.fit_transform(params)
samples_tsne = TSNE(n_components=2).fit_transform(reduced_features)
tsne_scatter = plt.scatter(samples_tsne[:, 0], samples_tsne[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show(tsne_scatter)
