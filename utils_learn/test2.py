import torch

outputs =torch.tensor([[0.1,0.2],
                       [0.3,0.4]])
print(outputs.argmax(1)) #1的时候是横向比，0的时候是纵向比较

preds = outputs.argmax(1)
targets = torch.tensor([0,1])

print(preds==targets)
print((preds==targets).sum())