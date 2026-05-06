import torch
from prepare_network import cnn_network_contrastive  

# =========================
# Config
# =========================
dropout_type = "drop1d"
p1 = 0.1
p2 = 0.1
p3 = 0.1
embedding_dim = 320

BATCH_SIZE = 4  # test batch size

path = "/storage/doprakah/CLOCS/SecondaryHDD/Contrastive Learning Results/CMSC/mimiciv/leads_['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']/embedding_320/seed0/SecondaryHDD/Contrastive Learning Results/CMSC/mimiciv/leads_['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']/embedding_320/seed0/pretrained_weight"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

# =========================
# Build model
# =========================
model = cnn_network_contrastive(
    dropout_type=dropout_type,
    p1=p1,
    p2=p2,
    p3=p3,
    embedding_dim=embedding_dim
).to(device)

model.eval()

print("✅ Model created")
print(model)

# =========================
# Dummy input
# Expected: (B, 2500, 12, 2)
# =========================
x = torch.randn(BATCH_SIZE, 2500, 12, 2, device=device)

# =========================
# Forward BEFORE loading weights
# =========================
with torch.no_grad():
    emb_before = model.get_embedding(x)

print("✅ Forward pass before loading weights OK")
print("Embedding shape (before):", emb_before.shape)

# =========================
# Load pretrained weights
# =========================
def strip_compile_prefix(state_dict):
    return {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

state_dict = torch.load(path, map_location=device)
state_dict = strip_compile_prefix(state_dict)

missing, unexpected = model.load_state_dict(state_dict, strict=True)


print("✅ Weights loaded")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# =========================
# Forward AFTER loading weights
# =========================
with torch.no_grad():
    emb_after = model.get_embedding(x)

print("✅ Forward pass after loading weights OK")
print("Embedding shape (after):", emb_after.shape)

# =========================
# Check embeddings changed
# =========================
diff = torch.mean(torch.abs(emb_after - emb_before)).item()
print(f"Mean absolute difference (before vs after): {diff:.6f}")

assert diff > 1e-6, "❌ Weights did NOT change the output!"

print("✅ Weights successfully affect model output")

# =========================
# Full forward() test
# =========================
with torch.no_grad():
    output = model(x)

print("✅ Full forward() successful")
print("Output shape:", output.shape)

# =========================
# Sanity check: parameter stats
# =========================
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    break
