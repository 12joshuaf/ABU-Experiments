import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from collections import deque

ABU_ITERATIONS = 30


# Adaptive Blending Unit with 4 activations: ReLU, Tanh, ELU, Swish
# When one weight reaches 0.85, it becomes 1.0 and others become 0.0 (hard freeze)
class AdaptiveBlendingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = False
        self.hard_frozen = False  # When one activation dominates (>= 0.85)
        self.dominant_idx = None

        # Four blending parameters for ReLU, Tanh, ELU, Swish
        self.weights = nn.Parameter(torch.ones(4) / 4)  # Initialize equally

        # Track parameter history for freezing decision
        self.param_history = deque(maxlen=ABU_ITERATIONS)

    def forward(self, x):
        # Check if input is 2D (fully connected) or 4D (convolutional)
        is_2d = x.dim() == 2

        # If hard frozen, use only the dominant activation
        if self.hard_frozen and self.dominant_idx is not None:
            if self.dominant_idx == 0:  # ReLU
                return F.relu(x)
            elif self.dominant_idx == 1:  # Tanh
                return torch.tanh(x)
            elif self.dominant_idx == 2:  # ELU
                return F.elu(x)
            elif self.dominant_idx == 3:  # Swish
                return x * torch.sigmoid(x)

        # Apply softmax to ensure weights sum to 1
        if is_2d:
            w = F.softmax(self.weights, dim=0).view(1, 1, 4)
        else:
            w = F.softmax(self.weights, dim=0).view(1, 1, 4, 1, 1)

        # Compute all four activations
        if is_2d:
            relu_out = F.relu(x).unsqueeze(2)
            tanh_out = torch.tanh(x).unsqueeze(2)
            elu_out = F.elu(x).unsqueeze(2)
            swish_out = (x * torch.sigmoid(x)).unsqueeze(2)
        else:
            relu_out = F.relu(x).unsqueeze(2)
            tanh_out = torch.tanh(x).unsqueeze(2)
            elu_out = F.elu(x).unsqueeze(2)
            swish_out = (x * torch.sigmoid(x)).unsqueeze(2)

        # Stack activations
        activations = torch.cat([relu_out, tanh_out, elu_out, swish_out], dim=2)

        # Weighted blend
        output = (activations * w).sum(dim=2)

        return output

    def get_weight_values(self):
        # Return current weight values (after softmax)
        return F.softmax(self.weights, dim=-1).detach()

    def check_and_freeze(self, threshold=0.005):
        # Check if weight values have stabilized and freeze if so
        # Also check if any weight >= 0.85 for hard freeze
        if self.frozen:
            return True

        current_weights = self.get_weight_values()

        # Check for hard freeze condition (any weight >= 0.85)
        max_weight = torch.max(current_weights).item()
        if max_weight >= 0.85:
            self.hard_freeze(current_weights)
            return True

        self.param_history.append(current_weights.cpu().numpy())

        # Need at least ABU_ITERATIONS iterations to check stability
        if len(self.param_history) < ABU_ITERATIONS:
            return False

        # Calculate change over last ABU_ITERATIONS iterations
        weight_array = np.array(list(self.param_history))
        max_change = np.max(np.abs(weight_array[-1] - weight_array[0]))

        if max_change < threshold:
            # Check again if any weight is >= 0.85 before normal freeze
            if max_weight >= 0.85:
                self.hard_freeze(current_weights)
            else:
                self.freeze()
            return True

        return False

    def hard_freeze(self, current_weights):
        """Hard freeze: Set dominant weight to 1.0, others to 0.0"""
        self.frozen = True
        self.hard_frozen = True
        self.dominant_idx = torch.argmax(current_weights).item()

        # Set weights: dominant = large value, others = very small
        # Use raw weights (before softmax) to create near one-hot after softmax
        with torch.no_grad():
            new_weights = torch.full_like(self.weights, -10.0)  # Very negative
            new_weights[self.dominant_idx] = 10.0  # Very positive
            self.weights.copy_(new_weights)

        self.weights.requires_grad = False

        activation_names = ['ReLU', 'Tanh', 'ELU', 'Swish']
        weight_vals = self.get_weight_values().cpu().numpy()

        print(f"  HARD FROZEN ABU - {activation_names[self.dominant_idx]} reached >= 0.85")
        print(f"    Final weights: ReLU={weight_vals[0]:.4f}, Tanh={weight_vals[1]:.4f}, "
              f"ELU={weight_vals[2]:.4f}, Swish={weight_vals[3]:.4f}")

    def freeze(self):
        """Normal freeze: Stop training but keep current weight distribution"""
        self.frozen = True
        self.weights.requires_grad = False
        weight_vals = self.get_weight_values().cpu().numpy()

        print(f"  Frozen ABU with weights - ReLU: {weight_vals[0]:.4f}, "
              f"Tanh: {weight_vals[1]:.4f}, ELU: {weight_vals[2]:.4f}, Swish: {weight_vals[3]:.4f}")

    def unfreeze(self):
        self.frozen = False
        self.hard_frozen = False
        self.dominant_idx = None
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
            hard_frozen_count = sum(1 for abu in model.get_all_abus() if abu.hard_frozen)
            print(f'  Batch {i + 1}, Loss: {running_loss / 100:.3f}, '
                  f'Acc: {100. * correct / total:.2f}%, Frozen ABUs: {frozen_count}/5 '
                  f'(Hard: {hard_frozen_count})')
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
        hard_frozen_count = sum(1 for abu in model.get_all_abus() if abu.hard_frozen)
        print(f'Train Acc: {100. * train_acc:.2f}%, Val Acc: {100. * val_acc:.2f}%, '
              f'Val Loss: {val_loss:.3f}, Frozen ABUs: {frozen_count}/5 (Hard: {hard_frozen_count})')

        # Display current alpha values every 5 epochs
        if (epoch + 1) % 5 == 0:
            print('\nCurrent ABU weights (ReLU, Tanh, ELU, Swish):')
            for i, abu in enumerate(model.get_all_abus()):
                weight_vals = abu.get_weight_values()
                if abu.hard_frozen:
                    status = "HARD FROZEN"
                elif abu.frozen:
                    status = "FROZEN"
                else:
                    status = "active"
                print(f'  Layer {i + 1}: ReLU={weight_vals[0].item():.4f}, '
                      f'Tanh={weight_vals[1].item():.4f}, '
                      f'ELU={weight_vals[2].item():.4f}, '
                      f'Swish={weight_vals[3].item():.4f} [{status}]')

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
    activation_names = ['ReLU', 'Tanh', 'ELU', 'Swish']
    for i, abu in enumerate(model.get_all_abus()):
        weight_vals = abu.get_weight_values()
        if abu.hard_frozen:
            status = "HARD FROZEN"
            dominant = activation_names[abu.dominant_idx]
            print(f'Layer {i + 1}: {dominant} = 1.0 (others = 0.0) [{status}]')
        elif abu.frozen:
            status = "FROZEN"
            print(f'Layer {i + 1}: ReLU={weight_vals[0].item():.4f}, '
                  f'Tanh={weight_vals[1].item():.4f}, '
                  f'ELU={weight_vals[2].item():.4f}, '
                  f'Swish={weight_vals[3].item():.4f} [{status}]')
        else:
            status = "ACTIVE"
            print(f'Layer {i + 1}: ReLU={weight_vals[0].item():.4f}, '
                  f'Tanh={weight_vals[1].item():.4f}, '
                  f'ELU={weight_vals[2].item():.4f}, '
                  f'Swish={weight_vals[3].item():.4f} [{status}]')


if __name__ == '__main__':
    main()