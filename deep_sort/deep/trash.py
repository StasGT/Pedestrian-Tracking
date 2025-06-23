import torch
from resnet import resnet18

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = resnet18(reid=True).to(device)
model.load_state_dict(torch.load("checkpoint/resnet18-5c106cde.pth"))
print(model)
del model
torch.cuda.empty_cache()