import pickle
import torch
import argparse
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv #GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--hc', type=int, default = 16, help = "number of hidden channels")
parser.add_argument('--lr', type=float, default = 0.01, help = "learning rate")
parser.add_argument('--decay', type=float, default = 5e-4, help = "decay rate")
parser.add_argument('--epochs', type=int, default = 1000, help = "epochs")
args = parser.parse_args()

#load pickle dataset
file = open('cora_data.pkl', 'rb')
dataset = pickle.load(file)
file.close()

# storing the graph in the data variable
data = dataset[0]  

# some statistics about the graph.
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Is undirected: {data.is_undirected()}')


# GAT model
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GATConv(dataset.num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x

# Initialize model
model = GAT(hidden_channels=args.hc)

# Use CPU
device = torch.device("cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = args.lr
decay = args.decay
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

losses = []

for epoch in range(0, args.epochs+1):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'***** Evaluating the test dataset ***** ')
print(f'Test Accuracy: {test_acc:.4f}')
