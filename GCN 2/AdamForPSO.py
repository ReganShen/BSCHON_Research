import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self,dataset):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)


        return out, h


class AdamTimeOnParticlePos():
    def __init__(self,dataset):
        torch.set_grad_enabled(True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataset = dataset
        self.model = GCN(self.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.02)




    def accuracy(self,pred_y, y):
        return (pred_y == y).sum() / len(y)

    def train(self,data):
        self.optimizer.zero_grad()  # Clear gradients.

        out, h = self.model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = self.criterion(out, data.y)
        loss.backward()  # Derive gradients.

        acc = self.accuracy(out.argmax(dim=1), data.y)
        self.optimizer.step()  # Update parameters based on gradients.


        # print(loss)
        return loss, h, acc

    def changeWeights(self,newWeights):
        for i in range(len(newWeights)):
            for j in range(len(self.optimizer.param_groups[i]['params'])):
                self.optimizer.param_groups[i]['params'][j].data = newWeights[i]['params'][j].data

    def itsTimeIthink(self):
        data = self.dataset[0]
        l = ""
        for epoch in range(100):
            loss,h, acc = self.train(data)
            l = loss
        return self.optimizer.param_groups