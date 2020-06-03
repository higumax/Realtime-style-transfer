import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from network import Model, LossNet
import numpy as np
from PIL import Image

from sys import argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_grammat(x):
    s, c, h, w = x.size()
    x = x.view(s, c, h * w)
    xT = x.transpose(1, 2)
    gram = x.bmm(xT)
    return gram / (c * h * w)

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
train_data = ImageFolder("./data", transform=preprocess)
train_data_loader = DataLoader(train_data, batch_size, shuffle=True)

# Load test image
test_img = Image.open("./img/monalisa.jpg")
test_img = preprocess(test_img)
test_img = torch.unsqueeze(test_img, 0).to(device)

# Model, optimizer, loss function
model = Model().to(device)
if len(argv) >= 2:
    model.load_state_dict(torch.load(argv[1]))
    print("Load done.")

lossnet = LossNet().to(device)
loss_mse = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Load a style image
style_img = Image.open("./img/starry_night.jpg")
style_img = preprocess(style_img)
style = style_img.repeat(batch_size, 1, 1, 1).to(device)
style_s, _ = lossnet(style)

for epoch in range(max_epoch):
    for i, data in enumerate(train_data_loader, 0):

        inputs, _ = data
        inputs = inputs.to(device)
        #print(inputs.shape, x.shape)

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs_s, outputs_f = lossnet(outputs)
        _, contents_f = lossnet(inputs)

        loss_feature = 0.0
        for x, y in zip(outputs_f, contents_f):
            loss_feature += loss_mse(x, y)

        loss_style = 0.0
        for x, y in zip(outputs_s, style_s):
            loss_style += loss_mse(to_grammat(x), to_grammat(y))

        diff_hori = torch.sum(torch.abs(outputs[:,:,:,1:] - outputs[:,:,:,:-1]))
        diff_vert = torch.sum(torch.abs(outputs[:,:,1:,] - outputs[:,:,:-1,:]))
        loss_tv = diff_hori + diff_vert

        total_loss = loss_feature + 1e5 * loss_style + 1e-7 * loss_tv
        total_loss = torch.clamp(total_loss, 0, 1e5)
        total_loss.backward()

        optimizer.step()

        print(f"[epoch {epoch+1:2d}/{max_epoch}, data {i:3d}/{len(train_data_loader)}] loss: {total_loss.item():.3f}")

    test_output = model(test_img)
    test_output = test_output.clone().detach().to("cpu").numpy()
    test_output = test_output[0] * np.array([[[0.229]], [[0.224]], [[0.225]]]) + np.array([[[0.485]], [[0.456]], [[0.406]]])
    test_output = (test_output.transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)

    output_img = Image.fromarray(test_output)
    output_img.save(f'epoch{epoch+1:02d}.jpg')

    torch.save(model.state_dict(), f"epoch{epoch+1:02d}.w")

