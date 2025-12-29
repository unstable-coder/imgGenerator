import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 64
BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 80
LR = 0.0001

DATA_DIR = "img_align_celeba/img_align_celeba"
ATTR_CSV = "list_attr_celeba.csv"
OUT_DIR = "generated"
os.makedirs(OUT_DIR, exist_ok=True)

ATTRIBUTES = ["Smiling", "Male"]

# ======================
# LOAD ATTRIBUTES
# ======================
attr_df = pd.read_csv(ATTR_CSV)
attr_df.rename(columns={attr_df.columns[0]: "image_id"}, inplace=True)

for col in ATTRIBUTES:
    attr_df[col] = pd.to_numeric(attr_df[col], errors="coerce")

attr_df.replace(-1, 0, inplace=True)
attr_df = attr_df[["image_id"] + ATTRIBUTES]
attr_df.dropna(inplace=True)

# ======================
# DATASET
# ======================
class CelebDataset(Dataset):
    def __init__(self, image_dir, attr_df, transform):
        self.image_dir = image_dir
        self.attr_df = attr_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        row = self.attr_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        attrs = torch.tensor(
            row[ATTRIBUTES].astype("float32").values,
            dtype=torch.float32
        )
        return image, attrs

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

dataset = CelebDataset(DATA_DIR, attr_df, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=2, pin_memory=True)

# ======================
# GENERATOR (DCGAN)
# ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Linear(len(ATTRIBUTES), LATENT_DIM)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, attrs):
        label_embed = self.label_emb(attrs)
        x = z + label_embed
        x = x.unsqueeze(2).unsqueeze(3)
        return self.net(x)

# ======================
# DISCRIMINATOR (DCGAN)
# ======================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Linear(len(ATTRIBUTES),
                                   IMAGE_SIZE * IMAGE_SIZE)

        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def forward(self, img, attrs):
        label = self.label_emb(attrs).view(-1, 1,
                                           IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([img, label], dim=1)
        return self.net(x).view(-1, 1)

# ======================
# INIT
# ======================
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# ======================
# TRAINING
# ======================
for epoch in range(EPOCHS):
    for imgs, attrs in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, attrs = imgs.to(DEVICE), attrs.to(DEVICE)
        b = imgs.size(0)

        real = torch.full((b, 1), 0.9, device=DEVICE)  # label smoothing
        fake = torch.zeros(b, 1, device=DEVICE)

        # ---- Train Discriminator ----
        z = torch.randn(b, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z, attrs)

        d_real = criterion(D(imgs, attrs), real)
        d_fake = criterion(D(fake_imgs.detach(), attrs), fake)
        d_loss = (d_real + d_fake) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---- Train Generator ----
        g_loss = criterion(D(fake_imgs, attrs), real)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    save_image(fake_imgs[:25],
               f"{OUT_DIR}/epoch_{epoch+1}.png",
               normalize=True)

    print(f"Epoch {epoch+1} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# ======================
# SAVE MODEL
# ======================
torch.save({
    "generator": G.state_dict(),
    "latent_dim": LATENT_DIM,
    "attributes": ATTRIBUTES,
    "image_size": IMAGE_SIZE
}, "generator.pth")

print("✅ Training complete. Model saved as generator.pth")
