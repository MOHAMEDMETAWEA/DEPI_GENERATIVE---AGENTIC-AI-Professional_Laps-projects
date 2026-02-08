
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# ===============================
# Configuration
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

BATCH_SIZE = 128
EPOCHS = 15
LR = 2e-4
T = 500  # Timesteps
Beta_start = 1e-4
Beta_end = 0.02
MODEL_STATUS_FILE = "training_status.txt"
MODEL_SAVE_PATH = "diffusion_unet_mnist.pth"

# ===============================
# Data Loading
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

os.makedirs("./data", exist_ok=True)
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# Diffusion Schedule
# ===============================
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

betas = make_beta_schedule(T, Beta_start, Beta_end).to(device)
alphas = (1.0 - betas).to(device)
alpha_bars = torch.cumprod(alphas, dim=0).to(device)

def q_sample(x0, t, eps):
    """
    x0: [B, 1, 28, 28]
    t: [B]
    eps: [B, 1, 28, 28]
    """
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    signal = torch.sqrt(a_bar) * x0
    noise = torch.sqrt(1.0 - a_bar) * eps
    return signal + noise

def sinusoidal_time_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

# ===============================
# U-Net Architecture
# ===============================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # Fixed: Added out_features=out_ch
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.conv1(x))
        time_added = self.time_proj(t_emb).view(-1, h.size(1), 1, 1)
        h = h + time_added
        h = F.silu(self.conv2(h))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, base=128, time_dim=128, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, base, kernel_size=3, padding=1)

        self.rb1 = ResBlock(base, base, time_dim)
        self.down1 = Down(base)

        self.rb2 = ResBlock(base, base * 2, time_dim)
        self.down2 = Down(base * 2)

        self.rb3 = ResBlock(base * 2, base * 2, time_dim)

        self.mid1 = ResBlock(base * 2, base * 4, time_dim)
        self.mid2 = ResBlock(base * 4, base * 4, time_dim)
        self.mid3 = ResBlock(base * 4, base * 2, time_dim)

        self.up1 = Up(base * 2)
        self.rb4 = ResBlock(base * 4, base * 2, time_dim)

        self.up2 = Up(base * 2)
        self.rb5 = ResBlock(base * 3, base, time_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, kernel_size=3, padding=1)

    def forward(self, x, t, y=None):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        if self.num_classes is not None:
            t_emb = t_emb + self.label_emb(y)

        x1 = self.rb1(self.in_conv(x), t_emb)
        x2 = self.rb2(self.down1(x1), t_emb)
        x3 = self.rb3(self.down2(x2), t_emb)

        h = self.mid1(x3, t_emb)
        h = self.mid2(h, t_emb)
        h = self.mid3(h, t_emb)

        h = self.up1(h)
        h = self.rb4(torch.cat([h, x2], dim=1), t_emb)

        h = self.up2(h)
        h = self.rb5(torch.cat([h, x1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))

# ===============================
# Training
# ===============================
def train():
    print("Setting up model...")
    # Conditional generation (digits 0-9)
    use_conditional = True
    model = UNet(
        in_ch=1,
        base=128,
        time_dim=128,
        num_classes=(10 if use_conditional else None)
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    
    for ep in range(1, EPOCHS + 1):
        epoch_losses = []
        for i, (x0, y) in enumerate(train_loader):
            x0 = x0.to(device)
            y = y.to(device)

            # 1) Sample random timesteps
            t = torch.randint(0, T, (x0.size(0),), device=device)

            # 2) Gaussian noise
            eps = torch.randn_like(x0)

            # 3) Forward diffusion
            xt = q_sample(x0, t, eps)

            # 4) Predict noise
            if use_conditional:
                eps_pred = model(xt, t, y=y)
            else:
                eps_pred = model(xt, t)

            # 5) Compute Loss
            loss = loss_fn(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())
            
            if i % 100 == 0:
                print(f"Epoch {ep} | Step {i} | Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {ep} | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint per epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    print("Training finished.")

if __name__ == "__main__":
    train()
