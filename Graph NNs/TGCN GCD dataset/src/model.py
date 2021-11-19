import torch
from torch import nn
from torch.nn import functional as F


class TGCN(nn.Module):
    def __init__(self, in_channels, emb_dims, in_dims, out_dims, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.emb_dims = emb_dims
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_classes = num_classes

        self.adj_out_dims = 256

        self.cnn = CNN_Module(in_channels, emb_dims)
        self.gcn1 = GCN_Layer(in_dims, in_dims)
        self.gcn2 = GCN_Layer(in_dims, out_dims)

        self.adjc_fc = nn.Linear(emb_dims, self.adj_out_dims)  # En el paper
        self.features_fc = nn.Linear(emb_dims, in_dims)

        self.fc1 = nn.Linear(emb_dims + out_dims, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, get_embeddings=False):
        embeddings = self.cnn(x)  # DEEP FEATURES

        embeddings, out = self.get_gcn_embeddings(x)
        out = torch.cat([embeddings, out], dim=-1)
        _out = F.relu(self.fc1(out))
        out = self.fc2(_out)

        if not get_embeddings:
            return out
        else:
            return (
                out,
                embeddings,
                _out,
            )  # prediction, graph_embeddings, final_embeddings

    def get_gcn_embeddings(self, x):
        embeddings = self.cnn(x)  # bs, deep_features_h

        # Adjacency matrix
        adj_linear = self.adjc_fc(embeddings)
        adjacency_matrix = F.softmax(adj_linear, dim=-1) @ (
            F.softmax(adj_linear, dim=-1).transpose(0, 1)
        )
        adjacency_matrix = torch.div(
            adjacency_matrix, torch.linalg.norm(adjacency_matrix, dim=-1)
        )

        # Feature matrix
        feature_matrix = self.features_fc(embeddings)

        # GCN
        out = F.relu(self.gcn1(adjacency_matrix, feature_matrix))
        out = self.gcn2(adjacency_matrix, out)

        return embeddings, out

    def get_final_embeddings(self, x):
        embeddings, out = self.get_gcn_embeddings(x)
        out = cat_tensor = torch.cat([embeddings, out], dim=-1)
        out = F.relu(self.fc1(out))

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class CNN_Module(nn.Module):
    def __init__(self, in_channels, emb_dims):
        super().__init__()
        self.out_channels = emb_dims

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
        self.mp4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(1024, self.out_channels, kernel_size=3, padding=1)

        self.av1 = nn.AvgPool2d((32, 32))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp4(x)
        x = F.relu(self.conv5(x))
        x = self.av1(x)
        x = x.reshape(-1, self.out_channels)  # Deep Features
        return x


class GCN_Layer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()

        self.lin = nn.Linear(in_dims, out_dims)

    def forward(self, adjacency_mat, feature_mat):
        x = adjacency_mat @ feature_mat  # N, in_dims
        x = F.leaky_relu(self.lin(x))
        return x


if __name__ == "__main__":
    tgcn = TGCN(
        in_channels=3,
        emb_dims=2048,  # Deep features
        in_dims=256,
        out_dims=512,  # Graph features
        num_classes=7,
    )
    img = torch.randn(5, 3, 256, 256)
    print(tgcn(img).shape)
