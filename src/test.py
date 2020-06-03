import numpy as np
from PIL import Image
from sys import argv
import torch
from torchvision import transforms
from network import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter for training
max_epoch = 100
batch_size = 10

# Set dataloader
preprocess = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load test image
print(argv[1])
test_img = Image.open(argv[1])
test_img = preprocess(test_img)
test_img = torch.unsqueeze(test_img, 0).to(device)

# Model, optimizer, loss function
model = Model().to(device)
model.load_state_dict(torch.load(argv[2]))

test_output = model(test_img)
test_output = test_output.clone().detach().to("cpu").numpy()
test_output = test_output[0] * np.array([[[0.229]], [[0.224]], [[0.225]]]) + np.array([[[0.485]], [[0.456]], [[0.406]]])
test_output = (test_output.transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)

output_img = Image.fromarray(test_output)
output_img.save(f'stylized.jpg')

