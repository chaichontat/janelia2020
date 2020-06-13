import torchvision.models as models
from torchvision import transforms
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#%%
print('Loading data.')
raw = np.load('superstim.npz')
img = raw['img']

img = np.transpose(img[:, 28:100, :], (2, 0, 1))[:, np.newaxis, :, :]
img = np.repeat(img, 3, axis=1)
img = torch.Tensor(img)
#%%

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

out = np.zeros([30000, 1000])

resnet = models.resnet50(pretrained=True, progress=True).to(device)
resnet.eval()
u = torch.Tensor(np.zeros([100, 3, 224, 224])).to(device)

with torch.no_grad():
    for k in range(300):
        print(k)
        for i in range(100):
            u[i, ...] = transform(img[k*100 + i, ...])

        out[k*100:(k+1)*100] = resnet(u).cpu().numpy()

np.savetxt('fuck.txt', out)