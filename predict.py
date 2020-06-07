import torch
import json
from PIL import Image
from torchvision.models import resnet34
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
from torch.nn.functional import softmax

img = Image.open('./inputs/kitten.jpg')

with open('./index_to_name.json') as f:
    index2name = json.load(f)

prep = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
x = prep(img)

model = resnet34(pretrained=True)
model.eval()

with torch.no_grad():
    preds = model(x.unsqueeze(0))
    probs = softmax(preds, dim=1)
    pred = probs.argmax()
    print(f"Predicted {index2name[str(pred.item())][1]} with id {pred.item()}")
