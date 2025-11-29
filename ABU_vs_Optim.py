import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

ABU_ITERATIONS = 30


# Optimized ABU with freezing (4 activations: ReLU, Tanh, ELU, Swish)
# FREEZES training when threshold is met
# OPTIMIZES computation when dominant weight >= 0.85
class OptimizedABU(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = False
        self.num_activations = 4

        # Learnable blending weights for 4 activations
        self.weights = nn.Parameter(torch.ones(4) / 4)

        # Track parameter history for freezing decision
        self.param_history = deque(maxlen=ABU_ITERATIONS)

        # Store frozen weights for display
        self.frozen_weights = None

        # Optimization: if dominant activation >= 0.85, only use that one
        self.use_single_activation = False
        self.dominant_activation_idx = None

    def forward(self, x):
        is_2d = x.dim() == 2

        # OPTIMIZED PATH: If dominant activation >= 0.85, only compute that one
        if self.use_single_activation and self.dominant_activation_idx is not None:
            if self.dominant_activation_idx == 0:  # ReLU
                return F.relu(x)
            elif self.dominant_activation_idx == 1:  # Tanh
                return torch.tanh(x)
            elif self.dominant_activation_idx == 2:  # ELU
                return F.elu(x)
            elif self.dominant_activation_idx == 3:  # Swish
                return x * torch.sigmoid(x)

        # Standard adaptive blending
        if is_2d:
            w = F.softmax(self.weights, dim=0).view(1, 1, 4)
        else:
            w = F.softmax(self.weights, dim=0).view(1, 1, 4, 1, 1)

        # Compute 4 activations: ReLU, Tanh, ELU, Swish
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

        # Stack and blend
        activations = torch.cat([relu_out, tanh_out, elu_out, swish_out], dim=2)
        output = (activations * w).sum(dim=2)

        return output

    def get_weight_values(self):
        return F.softmax(self.weights, dim=-1).detach()

    def check_and_freeze(self, threshold=0.005):
        if self.frozen:
            return True

        current_weights = self.get_weight_values()
        self.param_history.append(current_weights.cpu().numpy())

        if len(self.param_history) < ABU_ITERATIONS:
            return False

        # Calculate change over last ABU_ITERATIONS iterations
        weight_array = np.array(list(self.param_history))
        max_change = np.max(np.abs(weight_array[-1] - weight_array[0]))

        if max_change < threshold:
            self.freeze()
            return True

        return False

    def freeze(self):
        # STOPS TRAINING: Set requires_grad=False
        self.frozen = True
        self.weights.requires_grad = False
        weight_vals = self.get_weight_values()

        # Store frozen weights for later display
        self.frozen_weights = weight_vals.cpu().numpy().copy()

        activation_names = ['ReLU', 'Tanh', 'ELU', 'Swish']
        dominant_idx = torch.argmax(weight_vals).item()
        dominant_weight = weight_vals[dominant_idx].item()

        # Check if dominant activation is >= 0.85
        if dominant_weight >= 0.85:
            self.use_single_activation = True
            self.dominant_activation_idx = dominant_idx
            print(f"  Frozen with {activation_names[dominant_idx]} dominant ({dominant_weight:.4f} >= 0.85)")
            print(f"  → OPTIMIZED: Using only {activation_names[dominant_idx]} for forward pass")
        else:
            print(f"  Frozen with {activation_names[dominant_idx]} dominant "
                  f"(ReLU={weight_vals[0]:.4f}, Tanh={weight_vals[1]:.4f}, "
                  f"ELU={weight_vals[2]:.4f}, Swish={weight_vals[3]:.4f})")
            print(f"  → Continues blending (dominant weight {dominant_weight:.4f} < 0.85)")


# Standard ABU that NEVER freezes training (5 activations: ReLU, Tanh, ELU, Swish, Identity)
# Continues training weights throughout entire training process
class StandardABU(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_activations = 5

        # Learnable blending weights for 5 activations
        self.weights = nn.Parameter(torch.ones(5) / 5)

    def forward(self, x):
        is_2d = x.dim() == 2

        # ALWAYS compute all 5 activations and continue training
        if is_2d:
            w = F.softmax(self.weights, dim=0).view(1, 1, 5)
        else:
            w = F.softmax(self.weights, dim=0).view(1, 1, 5, 1, 1)

        # Compute all 5 activations: ReLU, Tanh, ELU, Swish, Identity
        if is_2d:
            relu_out = F.relu(x).unsqueeze(2)
            tanh_out = torch.tanh(x).unsqueeze(2)
            elu_out = F.elu(x).unsqueeze(2)
            swish_out = (x * torch.sigmoid(x)).unsqueeze(2)
            identity_out = x.unsqueeze(2)  # Identity (no transformation)
        else:
            relu_out = F.relu(x).unsqueeze(2)
            tanh_out = torch.tanh(x).unsqueeze(2)
            elu_out = F.elu(x).unsqueeze(2)
            swish_out = (x * torch.sigmoid(x)).unsqueeze(2)
            identity_out = x.unsqueeze(2)  # Identity (no transformation)

        activations = torch.cat([relu_out, tanh_out, elu_out, swish_out, identity_out], dim=2)
        output = (activations * w).sum(dim=2)

        return output

    def get_weight_values(self):
        return F.softmax(self.weights, dim=-1).detach()

    def check_and_freeze(self, threshold=0.005):
        # NEVER freezes - this method exists for interface compatibility
        return False

    def freeze(self):
        # NEVER freezes
        pass


# Simple CNN for benchmarking
class BenchmarkNet(nn.Module):
    def __init__(self, abu_class):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.abu1 = abu_class()

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.abu2 = abu_class()

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.abu3 = abu_class()

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.abu4 = abu_class()
        self.fc2 = nn.Linear(256, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.abu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.abu2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.abu3(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.abu4(x)
        x = self.fc2(x)

        return x

    def get_all_abus(self):
        return [self.abu1, self.abu2, self.abu3, self.abu4]

    def check_and_freeze_abus(self, threshold=0.005):
        frozen_count = 0
        for abu in self.get_all_abus():
            if abu.check_and_freeze(threshold):
                frozen_count += 1
        return frozen_count


def benchmark_training(model, device, num_batches=200, batch_size=64):
    """Benchmark training speed"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy data
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    dummy_labels = torch.randint(0, 10, (batch_size,)).to(device)

    times = []
    frozen_counts = []
    trainable_params = []

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

    # Actual benchmark
    for i in range(num_batches):
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

        # Check for freezing every 2 batches
        if i % 2 == 1:
            model.check_and_freeze_abus()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        batch_time = time.time() - start_time
        times.append(batch_time)

        # Count frozen ABUs
        frozen_count = sum(1 for abu in model.get_all_abus() if hasattr(abu, 'frozen') and abu.frozen)
        frozen_counts.append(frozen_count)

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params.append(trainable)

    return times, frozen_counts, trainable_params


def benchmark_inference(model, device, num_batches=500, batch_size=64):
    """Benchmark inference speed"""
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    times = []

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Actual benchmark
    with torch.no_grad():
        for _ in range(num_batches):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            batch_time = time.time() - start_time
            times.append(batch_time)

    return times


def plot_results(optimized_train_times, standard_train_times,
                 optimized_frozen, standard_frozen,
                 optimized_trainable, standard_trainable,
                 optimized_inference_times, standard_inference_times):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training time comparison
    ax = axes[0, 0]
    ax.plot(optimized_train_times, label='Optimized ABU (4 acts, freezes)', alpha=0.7, linewidth=1.5)
    ax.plot(standard_train_times, label='Standard ABU (5 acts, never freezes)', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time per Batch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trainable parameters over time
    ax = axes[0, 1]
    ax.plot(optimized_trainable, label='Optimized ABU', alpha=0.7, linewidth=1.5)
    ax.plot(standard_trainable, label='Standard ABU', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Number of Trainable Parameters')
    ax.set_title('Trainable Parameters (shows when ABUs freeze)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative training time
    ax = axes[1, 0]
    optimized_cumulative = np.cumsum(optimized_train_times)
    standard_cumulative = np.cumsum(standard_train_times)
    ax.plot(optimized_cumulative, label='Optimized ABU', alpha=0.7, linewidth=1.5)
    ax.plot(standard_cumulative, label='Standard ABU', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cumulative Time (seconds)')
    ax.set_title('Cumulative Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inference time comparison (histogram)
    ax = axes[1, 1]
    ax.hist(optimized_inference_times, bins=50, alpha=0.5, label='Optimized (4 acts)', density=True, color='blue')
    ax.hist(standard_inference_times, bins=50, alpha=0.5, label='Standard (5 acts)', density=True, color='orange')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('abu_benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'abu_benchmark_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("ABU Speed Comparison Benchmark")
    print("=" * 80)
    print("\nOptimized ABU: 4 activations (ReLU, Tanh, ELU, Swish), FREEZES training")
    print("Standard ABU:  5 activations (ReLU, Tanh, ELU, Swish, Identity), NEVER freezes")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Training benchmark
    print("\n" + "-" * 80)
    print("TRAINING BENCHMARK")
    print("-" * 80)

    print("\n1. Benchmarking Optimized ABU (4 activations, with freezing)...")
    model_optimized = BenchmarkNet(OptimizedABU).to(device)
    opt_train_times, opt_frozen, opt_trainable = benchmark_training(model_optimized, device, num_batches=200)

    print("\n2. Benchmarking Standard ABU (5 activations, never freezes)...")
    model_standard = BenchmarkNet(StandardABU).to(device)
    std_train_times, std_frozen, std_trainable = benchmark_training(model_standard, device, num_batches=200)

    # Training statistics
    print("\n" + "-" * 80)
    print("TRAINING RESULTS")
    print("-" * 80)

    opt_mean = np.mean(opt_train_times) * 1000
    std_mean = np.mean(std_train_times) * 1000
    opt_total = np.sum(opt_train_times)
    std_total = np.sum(std_train_times)

    print(f"\nOptimized ABU (4 activations, freezes):")
    print(f"  Mean batch time: {opt_mean:.3f} ms")
    print(f"  Total time: {opt_total:.2f} s")
    print(f"  Final frozen ABUs: {opt_frozen[-1]}/4")
    print(f"  Final trainable params: {opt_trainable[-1]:,}")

    print(f"\nStandard ABU (5 activations, never freezes):")
    print(f"  Mean batch time: {std_mean:.3f} ms")
    print(f"  Total time: {std_total:.2f} s")
    print(f"  Final frozen ABUs: {std_frozen[-1]}/4")
    print(f"  Final trainable params: {std_trainable[-1]:,}")

    if std_total > opt_total:
        speedup_training = ((std_total - opt_total) / std_total) * 100
        print(f"\n✓ Training Speedup: {speedup_training:.2f}% faster (Optimized wins)")
        print(f"  Time saved: {std_total - opt_total:.2f} seconds")
    else:
        slowdown_training = ((opt_total - std_total) / std_total) * 100
        print(f"\n✗ Training Slowdown: {slowdown_training:.2f}% slower (Standard wins)")
        print(f"  Extra time: {opt_total - std_total:.2f} seconds")

    # Inference benchmark
    print("\n" + "-" * 80)
    print("INFERENCE BENCHMARK")
    print("-" * 80)

    print("\n3. Benchmarking Optimized ABU inference...")
    opt_inference_times = benchmark_inference(model_optimized, device, num_batches=500)

    print("4. Benchmarking Standard ABU inference...")
    std_inference_times = benchmark_inference(model_standard, device, num_batches=500)

    # Inference statistics
    print("\n" + "-" * 80)
    print("INFERENCE RESULTS")
    print("-" * 80)

    opt_inf_mean = np.mean(opt_inference_times) * 1000
    std_inf_mean = np.mean(std_inference_times) * 1000

    print(f"\nOptimized ABU (4 activations):")
    print(f"  Mean inference time: {opt_inf_mean:.3f} ms")
    print(f"  Throughput: {1000 / opt_inf_mean:.1f} batches/second")

    print(f"\nStandard ABU (5 activations):")
    print(f"  Mean inference time: {std_inf_mean:.3f} ms")
    print(f"  Throughput: {1000 / std_inf_mean:.1f} batches/second")

    if std_inf_mean > opt_inf_mean:
        speedup_inference = ((std_inf_mean - opt_inf_mean) / std_inf_mean) * 100
        print(f"\n✓ Inference Speedup: {speedup_inference:.2f}% faster (Optimized wins)")
    else:
        slowdown_inference = ((opt_inf_mean - std_inf_mean) / std_inf_mean) * 100
        print(f"\n✗ Inference Slowdown: {slowdown_inference:.2f}% slower (Standard wins)")

    # Final activation analysis
    print("\n" + "-" * 80)
    print("FINAL ACTIVATION WEIGHTS")
    print("-" * 80)

    print("\nOptimized ABU (4 activations: ReLU, Tanh, ELU, Swish):")
    activation_names_4 = ['ReLU', 'Tanh', 'ELU', 'Swish']
    for i, abu in enumerate(model_optimized.get_all_abus()):
        weights = abu.get_weight_values().cpu().numpy()
        dominant = np.argmax(weights)
        status = "FROZEN" if hasattr(abu, 'frozen') and abu.frozen else "ACTIVE"

        print(f"  Layer {i + 1}: [{status}]")
        print(f"    ReLU={weights[0]:.4f}, Tanh={weights[1]:.4f}, "
              f"ELU={weights[2]:.4f}, Swish={weights[3]:.4f}")
        print(f"    Dominant: {activation_names_4[dominant]} ({weights[dominant]:.4f})")

        # Show frozen weights if available
        if hasattr(abu, 'frozen_weights') and abu.frozen_weights is not None:
            fw = abu.frozen_weights
            print(f"    Frozen at: ReLU={fw[0]:.4f}, Tanh={fw[1]:.4f}, "
                  f"ELU={fw[2]:.4f}, Swish={fw[3]:.4f}")

    print("\nStandard ABU (5 activations: ReLU, Tanh, ELU, Swish, Identity - never frozen):")
    activation_names_5 = ['ReLU', 'Tanh', 'ELU', 'Swish', 'Identity']
    for i, abu in enumerate(model_standard.get_all_abus()):
        weights = abu.get_weight_values().cpu().numpy()
        dominant = np.argmax(weights)
        print(f"  Layer {i + 1}: [ACTIVE - continues training]")
        print(f"    ReLU={weights[0]:.4f}, Tanh={weights[1]:.4f}, ELU={weights[2]:.4f}, "
              f"Swish={weights[3]:.4f}, Identity={weights[4]:.4f}")
        print(f"    Dominant: {activation_names_5[dominant]} ({weights[dominant]:.4f})")

    # Plot results
    print("\n" + "-" * 80)
    print("Generating visualization...")
    plot_results(opt_train_times, std_train_times,
                 opt_frozen, std_frozen,
                 opt_trainable, std_trainable,
                 opt_inference_times, std_inference_times)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Optimized ABU: 4 activations (ReLU, Tanh, ELU, Swish), freezes training")
    print(f"Standard ABU:  5 activations (ReLU, Tanh, ELU, Swish, Identity), never freezes")
    print()
    if std_total > opt_total:
        print(f"✓ Training: Optimized is {((std_total - opt_total) / std_total) * 100:.2f}% FASTER")
    else:
        print(f"✗ Training: Optimized is {((opt_total - std_total) / std_total) * 100:.2f}% SLOWER")

    if std_inf_mean > opt_inf_mean:
        print(f"✓ Inference: Optimized is {((std_inf_mean - opt_inf_mean) / std_inf_mean) * 100:.2f}% FASTER")
    else:
        print(f"✗ Inference: Optimized is {((opt_inf_mean - std_inf_mean) / std_inf_mean) * 100:.2f}% SLOWER")
    print("=" * 80)


if __name__ == '__main__':
    main()