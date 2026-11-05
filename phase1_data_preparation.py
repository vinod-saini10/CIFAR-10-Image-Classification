
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# 1. CIFAR-10 class names
classes = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# 2. Normalization values for CIFAR-10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# 3. Training transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 4. Download and load dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True
)

print("Dataset loaded successfully!")
print("Total training images:", len(train_dataset))
print("Image size:", train_dataset[0][0].size)
print("Number of classes:", len(classes))

# 5. Display 10 random images
plt.figure(figsize=(12, 6))

random_indices = random.sample(range(len(train_dataset)), 10)

for i, index in enumerate(random_indices):
    image, label = train_dataset[index]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(classes[label])
    plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Demonstrate data augmentation
augmented_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=train_transform
)

plt.figure(figsize=(12, 6))

for i, index in enumerate(random_indices):
    image, label = augmented_dataset[index]

    # Undo normalization for display
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    image = image.clamp(0, 1)

    plt.subplot(2, 5, i + 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(classes[label])
    plt.axis('off')

plt.suptitle("Augmented CIFAR-10 Images")
plt.tight_layout()
plt.show()

print("Done- phase 1 ")
