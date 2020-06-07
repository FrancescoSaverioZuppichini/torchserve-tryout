import torch
from torchvision.models import resnet34

model = resnet34(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
# following issue https://github.com/pytorch/serve/issues/364 you need to explicitly set the model to .eval
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("./resnet34.pt")