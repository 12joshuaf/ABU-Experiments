import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from collections import deque

ABU_ITERATIONS = 30


# Adaptive Blending Unit - Single mode only
class AdaptiveBlendingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = False

        # Single set of three blending parameters
        self.weights = nn.Parameter(torch.ones(3) / 3)  # Initialize equally

        # Track parameter history for freezing decision
        self.param_history = deque(maxlen=30)

    def forward(self, x):
        # Check if input is 2D (fully connected) or 4D (convolutional)
        is_2d = x.dim() == 2

        # Apply softmax to ensure weights sum to 1
        if is_2d:
            w = F.softmax(self.weights, dim=0).view(1, 1, 3)
        else:
            w = F.softmax(self.weights, dim=0).view(1, 1, 3, 1, 1)

        # Compute all three activations
        if is_2d:
            relu_out = F.relu(x).unsqueeze(2)
            tanh_out = torch.tanh(x).unsqueeze(2)
            sigmoid_out = torch.sigmoid(x).unsqueeze(2)
        else:
            relu_out = F.relu(x).unsqueeze(2)
            tanh_out = torch.tanh(x).unsqueeze(2)
            sigmoid_out = torch.sigmoid(x).unsqueeze(2)

        # Stack activations
        activations = torch.cat([relu_out, tanh_out, sigmoid_out], dim=2)

        # Weighted blend
        output = (activations * w).sum(dim=2)

        return output

    def get_weight_values(self):
        # Return current weight values (after softmax)
        return F.softmax(self.weights, dim=-1).detach()

    def check_and_freeze(self, threshold=0.005):
        # Check if weight values have stabilized and freeze if so. Returns True if frozen, False otherwise.
        if self.frozen:
            return True

        current_weights = self.get_weight_values()
        self.param_history.append(current_weights.cpu().numpy())

        # Need at least 30 iterations to check stability
        if len(self.param_history) < ABU_ITERATIONS:
            return False

        # Calculate change over last 30 iterations
        weight_array = np.array(list(self.param_history))
        max_change = np.max(np.abs(weight_array[-1] - weight_array[0]))

        if max_change < threshold:
            self.freeze()
            return True

        return False

    def freeze(self):
        self.frozen = True
        self.weights.requires_grad = False
        weight_vals = self.get_weight_values().cpu().numpy()

        print(f"  Frozen ABU with weights - ReLU: {weight_vals[0]:.4f}, "
              f"Tanh: {weight_vals[1]:.4f}, Sigmoid: {weight_vals[2]:.4f}")

    def unfreeze(self):
        self.frozen = False
        self.weights.requires_grad = True


# CNN with ABU activations
class CIFAR10_ABU_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.abu1 = AdaptiveBlendingUnit()

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.abu2 = AdaptiveBlendingUnit()

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.abu3 = AdaptiveBlendingUnit()

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.abu4 = AdaptiveBlendingUnit()

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.abu5 = AdaptiveBlendingUnit()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.abu1(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.abu2(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.abu3(x)
        x = self.pool(x)

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.abu4(x)
        x = self.pool(x)

        # Fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.abu5(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_all_abus(self):
        # Return all ABU modules
        return [self.abu1, self.abu2, self.abu3, self.abu4, self.abu5]

    def check_and_freeze_abus(self, threshold=0.005):
        # Check all ABUs and freeze those that have stabilized
        frozen_count = 0
        for i, abu in enumerate(self.get_all_abus()):
            was_frozen = abu.frozen
            if abu.check_and_freeze(threshold):
                if abu.frozen and not was_frozen:  # Just frozen
                    print(f"Layer {i + 1} ABU frozen")
                frozen_count += 1
        return frozen_count


def train_epoch(model, trainloader, criterion, optimizer, device, epoch, abu_freeze_threshold=0.005):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Check ABU stability every 2 batches
        if i % 2 == 1:
            frozen_count = model.check_and_freeze_abus(threshold=abu_freeze_threshold)

        if i % 100 == 99:
            frozen_count = sum(1 for abu in model.get_all_abus() if abu.frozen)
            print(f'  Batch {i + 1}, Loss: {running_loss / 100:.3f}, '
                  f'Acc: {100. * correct / total:.2f}%, Frozen ABUs: {frozen_count}/5')
            running_loss = 0.0

    return correct / total


def validate(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total, val_loss / len(testloader)


def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.001
    abu_freeze_threshold = 0.005  # Threshold for freezing ABUs

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10
    print('Loading CIFAR-10 dataset...')
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=2)

    # Model, loss, optimizer
    print('Initializing model...')
    model = CIFAR10_ABU_Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    print('\nStarting training...')
    print('=' * 70)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 70)

        # Train
        train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch, abu_freeze_threshold)

        # Validate
        val_acc, val_loss = validate(model, testloader, criterion, device)

        frozen_count = sum(1 for abu in model.get_all_abus() if abu.frozen)
        print(f'Train Acc: {100. * train_acc:.2f}%, Val Acc: {100. * val_acc:.2f}%, '
              f'Val Loss: {val_loss:.3f}, Frozen ABUs: {frozen_count}/5')

        # Display current alpha values every 5 epochs
        if (epoch + 1) % 5 == 0:
            print('\nCurrent ABU weights (ReLU, Tanh, Sigmoid):')
            for i, abu in enumerate(model.get_all_abus()):
                weight_vals = abu.get_weight_values()
                status = "FROZEN" if abu.frozen else "active"
                print(f'  Layer {i + 1}: ReLU={weight_vals[0].item():.4f}, '
                      f'Tanh={weight_vals[1].item():.4f}, '
                      f'Sigmoid={weight_vals[2].item():.4f} [{status}]')

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'cifar10_abu_best.pth')
            print(f'  → Saved new best model (acc: {100. * best_acc:.2f}%)')

    print('\n' + '=' * 70)
    print(f'Training complete! Best validation accuracy: {100. * best_acc:.2f}%')

    # Final ABU statistics
    print('\nFinal ABU Status:')
    print('-' * 70)
    for i, abu in enumerate(model.get_all_abus()):
        weight_vals = abu.get_weight_values()
        status = "FROZEN" if abu.frozen else "ACTIVE"
        print(f'Layer {i + 1}: ReLU={weight_vals[0].item():.4f}, '
              f'Tanh={weight_vals[1].item():.4f}, '
              f'Sigmoid={weight_vals[2].item():.4f} [{status}]')


if __name__ == '__main__':
    main()
