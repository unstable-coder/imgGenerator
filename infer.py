import io
import base64
from PIL import Image

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD CHECKPOINT
# ======================
ckpt = torch.load("generator.pth", map_location=DEVICE)

LATENT_DIM = ckpt["latent_dim"]
ATTRIBUTES = ckpt["attributes"]
IMAGE_SIZE = ckpt["image_size"]

# ======================
# GENERATOR
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

G = Generator().to(DEVICE)
G.load_state_dict(ckpt["generator"])
G.eval()

# ======================
# GENERATE IMAGE
# ======================
def generate_from_input(user_input: list, variations: int = 1, expression: str = None):
    """
    Generate `variations` images for the same attribute vector.

    Args:
        user_input: list of attribute values (length == len(ATTRIBUTES))
        variations: number of different latent samples to generate

    Returns:
        List of base64-encoded PNG images (strings).
    """

    # Ensure attribute vector length matches model expectation
    attrs_list = list(user_input or [])
    if len(attrs_list) < len(ATTRIBUTES):
        attrs_list = attrs_list + [0] * (len(ATTRIBUTES) - len(attrs_list))
    elif len(attrs_list) > len(ATTRIBUTES):
        attrs_list = attrs_list[: len(ATTRIBUTES)]

    # Clamp variations
    try:
        variations = int(variations)
    except Exception:
        variations = 1
    variations = max(1, min(12, variations))

    # Expression controls latent sampling strength (hint only)
    expr = (expression or '').lower()
    # scales: lower -> less variation, higher -> stronger variation
    scale_map = {
        'happy': 0.6,   # light latent variation
        'neutral': 1.0, # normal
        'serious': 1.4, # medium
        'angry': 2.0     # strong
    }
    scale = float(scale_map.get(expr, 1.0))

    z = torch.randn(variations, LATENT_DIM).to(DEVICE) * scale
    attrs = torch.tensor([attrs_list] * variations, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        imgs = G(z, attrs)  # (variations, 3, H, W)

    out_images = []
    for i in range(imgs.size(0)):
        img = imgs[i].cpu()
        img = (img + 1) / 2  # from [-1,1] to [0,1]
        img = img.clamp(0, 1)
        np_img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        out_images.append(b64)

    return out_images