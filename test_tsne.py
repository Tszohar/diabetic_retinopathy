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

if __name__ == '__main__':

    batch_size = 256
    model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191106 (11:57:50.558861)_MULTI_CLASS_64_2HeadedResnet/model_epoch_33.pth"
    model_path = '/media/guy/Files 3/Tsofit/blindness detection/results/20191111 (16:28:16.076528)_MULTI_CLASS_64_HealthyAndSick/model_epoch_21.pth'
    colors = ['red', 'green', 'purple', 'black', 'cyan']


    train_dataset = BDDataset(csv_file=parameters.train_csv, data_dir=parameters.data_dir, transform=None)
    train2015_dataset = BDDataset(csv_file=parameters.train2015_csv, data_dir=parameters.data2015_dir, transform=None)
    dataset_combined = train_dataset + train2015_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    net = ResNet(num_outputs=5)
    model = torch.load(model_path)
    net.load_state_dict(model, strict=False)
    net.to(parameters.device)
    net.eval()
    torch.no_grad()
    torch.set_grad_enabled(False)

    for i_batch, sample_batched in enumerate(train_dataloader):
        params_vec = net.get_features(sample_batched['image'].to(parameters.device))
        if i_batch % 10 == 0:
            print("batch {}/{}".format(i_batch, len(train_dataloader)))

        if (i_batch == 0):
            params = np.array(params_vec.cpu())
            labels = np.array(sample_batched['diagnosis'])
        else:
            labels = np.concatenate((labels, np.array(sample_batched['diagnosis'])), axis=0)
            params = np.concatenate((params, params_vec.cpu()), axis=0)
    pca = PCA(n_components=10, svd_solver='full')
    reduced_features = pca.fit_transform(params)
    samples_tsne = TSNE(n_components=2).fit_transform(params)

    for label_idx in np.unique(labels):
        plt.plot(samples_tsne[labels == label_idx, 0], samples_tsne[labels == label_idx, 1], '.', label=f"{label_idx}")
        plt.legend()
    # tsne_scatter = plt.scatter(samples_tsne[:, 0], samples_tsne[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


