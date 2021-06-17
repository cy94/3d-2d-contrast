import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

x = torch.load('n_points.pth')
plt.hist(x, bins=100)
plt.ylabel('Number of scenes')
plt.xlabel('Number of points')
plt.savefig('n_points.png')

