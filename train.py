import torch
from torchvision import datasets, transforms
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from shared import get_noise_scheduler

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load datasets
train_dataset = datasets.ImageFolder('watermark_classifier_data/train', transform=transform)
validation_dataset = datasets.ImageFolder('watermark_classifier_data/validation', transform=transform)


noise_scheduler = get_noise_scheduler()

class DatasetWrapper:

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        clean_images, y = self.dataset[idx]
        return clean_images, y

train_dataset = DatasetWrapper(train_dataset)
validation_dataset = DatasetWrapper(validation_dataset)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


import timm
import torch.nn as nn

# Load a pre-trained model
model = timm.create_model('resnet34', pretrained=True, num_classes=2)

# Change the classifier to fit binary classification
model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming the final FC layer's name is 'fc'

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


import torch.optim as optim

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, train_loader, validation_loader, epochs=25):

    best_validation_accuracy = -1.
    if True:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Validation Accuracy: {100 * correct / total}%')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        # Validation step
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Validation Accuracy: {100 * correct / total}%')
            if best_validation_accuracy < (100 * correct / total):
                torch.save(model, "watermark_classifier.pt")
                best_validation_accuracy = (100 * correct / total)

# Call the training function
train_model(model, criterion, optimizer, train_loader, validation_loader)
