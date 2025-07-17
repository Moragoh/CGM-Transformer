import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CGMPredictor
from dataset import CGMDataset, collate_fn

# Optional performance tweaks
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Config
# -------------------------
learning_rate = 1e-4
num_iters = 50000
log_iters = 1000
warmup_iters = num_iters / 10
decay_iters = num_iters
final_lr = learning_rate / 10
accumulation_steps = 8

save_path = "results/"
os.makedirs(save_path, exist_ok=True)

# -------------------------
# Model
# -------------------------
model = CGMPredictor(
    n_embd=8*48,
    n_head=8,
    n_layer=3,
    dropout=0.3,
)

model = torch.compile(model)
model.to(device)

# Load the latest checkpoint if available
iteration = 1
#model.load_state_dict(torch.load(f'./results/model_iter_{iteration}.pth', map_location=device))
#if os.listdir(save_path):
#iteration = max(int(f.split('_')[-1][:-4]) for f in os.listdir(save_path) if f.startswith('model_iter_') and f.endswith('.pth'))
#model.load_state_dict(torch.load(f'{save_path}/model_iter_{iteration}.pth', map_location=device))

print(f'Trainable parameters: {model.num_params()}')

criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

# -------------------------
# Datasets and Loaders
# -------------------------
train_dataset = CGMDataset(file="./Datasets/Train", max_len=512, pred_time=90, augment=False, max_range = 1.5)
val_dataset = CGMDataset(file="./Datasets/Val", max_len=512, pred_time=90, augment=False,max_range = 1.5)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,  # No need for DistributedSampler
    collate_fn=collate_fn,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,  # No need for DistributedSampler
    collate_fn=collate_fn,
    drop_last=True,
)

# -------------------------
# Learning Rate Scheduler
# -------------------------
def get_lr(iteration):
    if iteration < warmup_iters:
        return learning_rate
    elif warmup_iters <= iteration <= warmup_iters + decay_iters:
        decay_progress = (iteration - warmup_iters) / decay_iters
        return learning_rate - (learning_rate - final_lr) * decay_progress
    else:
        return final_lr

# -------------------------
# Training Loop
# -------------------------
model.train()
optimizer.zero_grad()

running_loss = 0.0

while iteration <= num_iters:
    batch = next(iter(train_loader))

    # Update learning rate
    lr = get_lr(iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("DEBUG: Batch created.")

    # Move batch to device
    cgm = batch['cgm'].to(device)
    basal = batch['basal'].to(device)
    bolus = batch['bolus'].to(device)
    cgm_time = batch['cgm_time'].to(device)
    basal_time = batch['basal_time'].to(device)
    bolus_time = batch['bolus_time'].to(device)
    target_cgm = batch['target_cgm'].to(device)
    target_time = batch['target_time'].to(device)
    pred_time = batch['pred_time'].to(device)

    print("DEBUG: Batch moved to GPU.")


    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        print("DEBUG: Entering model forward pass...")
        output_cgm = model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
        print("DEBUG: Model forward pass complete.")
        loss = criterion(model.normalize_cgm(output_cgm), model.normalize_cgm(target_cgm))
        loss = loss / accumulation_steps

        print(f"\rIteration: {iteration}/{num_iters} | Loss: {(loss.item() * accumulation_steps):.6f}    |    ", end='', flush=True)

    if not torch.isnan(loss).any():
        running_loss += loss.item() * accumulation_steps
        iteration += 1

    if not torch.isnan(loss).any():
        loss.backward()

    if iteration % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

    if iteration % log_iters == 0:
        avg_train_loss = running_loss / log_iters
        print(f"\nIter {iteration}/{num_iters} - Train CGM Loss: {avg_train_loss:.6f}")
        running_loss = 0.0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                cgm = val_batch['cgm'].to(device)
                basal = val_batch['basal'].to(device)
                bolus = val_batch['bolus'].to(device)
                cgm_time = val_batch['cgm_time'].to(device)
                basal_time = val_batch['basal_time'].to(device)
                bolus_time = val_batch['bolus_time'].to(device)
                target_cgm = val_batch['target_cgm'].to(device)
                target_time = val_batch['target_time'].to(device)
                pred_time = val_batch['pred_time'].to(device)

                output_cgm = model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
                val_loss = criterion(model.normalize_cgm(output_cgm), model.normalize_cgm(target_cgm))
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Iter {iteration} - Validation CGM Loss: {avg_val_loss:.6f}")

        model_filename = f"{save_path}model_iter_{iteration}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved: {model_filename}")

        model.train()

print("Training and Validation complete!")
