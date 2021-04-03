from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import pickle

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

filename = 'cora_data.pkl'
outfile = open(filename,'wb')
pickle.dump(dataset,outfile)
outfile.close()
