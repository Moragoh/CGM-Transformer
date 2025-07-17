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
log_iters = 100
warmup_iters = num_iters / 10
decay_iters = num_iters
final_lr = learning_rate / 10
accumulation_steps = 8

save_path = "results/"
os.makedirs(save_path, exist_ok=True)

# -------------------------
# Model
# -------------------------
### --- Teacher Model ---
print("Loading Teacher Model...")
teacher_model = CGMPredictor(
    n_embd=8*48,  # 384
    n_head=8,
    n_layer=3,
    dropout=0.0,  # No dropout needed for inference
)

# Make a clean state_dict to get rid of errors caused by "residue" from previous runs
# 1. Load the state dictionary from the file into a variable
state_dict = torch.load("./results/model_iter_23000.pth", map_location=device)

# 2. Create a new, empty dictionary to hold the cleaned keys
from collections import OrderedDict
new_state_dict = OrderedDict()

# 3. Loop through the original state_dict and remove the "_orig_mod." prefix
for k, v in state_dict.items():
    if k.startswith('_orig_mod.'):
        name = k[10:] # remove `_orig_mod.`
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

# 4. Load the NEW, cleaned state_dict into the model
teacher_model.load_state_dict(new_state_dict)
teacher_model.to(device)
teacher_model.eval()  # IMPORTANT: Set teacher to evaluation mode. Freezes model so that it does not train anymore.
print(f'Teacher trainable parameters: {teacher_model.num_params()}')

### --- Student Model (The distilled model to be trained)
print("Initializing Student Model...")
student_model = CGMPredictor(
    n_embd=4*48,  # REDUCED: 192
    n_head=4,     # REDUCED
    n_layer=2,    # REDUCED
    dropout=0.3,  # Keep dropout for student training
)
student_model = torch.compile(student_model)
student_model.to(device)
print(f'Student trainable parameters: {student_model.num_params()}')

# -------------------------
# Loss and Optimizer
# -------------------------
criterion = nn.L1Loss()
# IMPORTANT: The optimizer now targets ONLY the STUDENT's parameters
optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=1e-3)

# We need to define a new loss which combines both loss regarding the grouth truth and one regarding the teacher model
def distillation_loss(student_output, teacher_output, true_labels, alpha=0.5):
    """
    Calculates the combined distillation loss.
    `alpha` balances the student's goal of matching the true labels vs. matching the teacher.
    """
    hard_loss = criterion(student_output, true_labels)      # Loss against the ground truth
    soft_loss = criterion(student_output, teacher_output)   # Loss against the teacher's output
    
    return (alpha * hard_loss) + ((1 - alpha) * soft_loss)

# We will start the new distillation training from iteration 1
iteration = 1

# -------------------------
# Datasets and Loaders
# -------------------------
train_dataset = CGMDataset(file="./Datasets/Train", max_len=512*2, pred_time=90, augment=False, max_range = 1.5)
val_dataset = CGMDataset(file="./Datasets/Val", max_len=512*10, pred_time=90, augment=False,max_range = 1.5)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,  # No need for DistributedSampler
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=4 if os.cpu_count() > 7 else 1, # Use most CPU cores
    # pin_memory=True, # Speed up data transfer to GPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,  # No need for DistributedSampler
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=4  if os.cpu_count() > 7 else 1, # Use most CPU cores
    # pin_memory=True, # Speed up data transfer to GPU
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
# -------------------------
# Training Loop
# -------------------------
student_model.train() # Make sure student is in training mode
optimizer.zero_grad()

running_loss = 0.0

while iteration <= num_iters:
    # Set learning rate
    lr = get_lr(iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    batch = next(iter(train_loader))

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

    # 1. Get teacher's predictions (soft labels). No gradients needed.
    with torch.no_grad():
        teacher_output_cgm = teacher_model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)

    # 2. Get student's predictions and calculate the combined distillation loss
    # We keep autocast disabled as we did to fix the segfault
    student_output_cgm = student_model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
    
    # Normalize all outputs before calculating loss
    norm_student_out = student_model.normalize_cgm(student_output_cgm)
    norm_teacher_out = student_model.normalize_cgm(teacher_output_cgm)
    norm_target = student_model.normalize_cgm(target_cgm)
    
    # Use our new distillation loss function
    loss = distillation_loss(norm_student_out, norm_teacher_out, norm_target, alpha=0.5)
    loss = loss / accumulation_steps

    print(f"\rIteration: {iteration}/{num_iters} | Loss: {(loss.item() * accumulation_steps):.6f}    |    ", end='', flush=True)

    if not torch.isnan(loss).any():
        running_loss += loss.item() * accumulation_steps
        loss.backward() # Calculate gradients for the student model

    if iteration % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
    
    iteration += 1
    
    # --- Start of Validation Block ---
    if iteration % log_iters == 0:
        # Debug Print 1: Indicate the start of the validation block
        print("\n--- Starting Validation Block ---")

        avg_train_loss = running_loss / log_iters
        print(f"Iter {iteration}/{num_iters} - Distillation Train Loss: {avg_train_loss:.6f}")
        running_loss = 0.0

        # --- Validation: Evaluate the STUDENT model on its own ---
        student_model.eval()  # Set the student to evaluation mode
        # Debug Print 2: Confirm student model is in eval mode
        print("Student model set to eval mode.")

        val_losses = []
        val_batch_counter = 0 # Initialize a counter to track validation batches

        with torch.no_grad():
            for val_batch in val_loader:
                val_batch_counter += 1
                # Debug Print 3: Show progress through validation batches
                # This uses '\r' to overwrite the same line, making it less spammy
                print(f"\rProcessing validation batch {val_batch_counter}/{len(val_loader)}...", end='', flush=True)

                cgm = val_batch['cgm'].to(device)
                basal = val_batch['basal'].to(device)
                bolus = val_batch['bolus'].to(device)
                cgm_time = val_batch['cgm_time'].to(device)
                basal_time = val_batch['basal_time'].to(device)
                bolus_time = val_batch['bolus_time'].to(device)
                target_cgm = val_batch['target_cgm'].to(device)
                target_time = val_batch['target_time'].to(device)
                pred_time = val_batch['pred_time'].to(device)

                # Get student's output ONLY
                # If the hang occurs here, it's during the forward pass of the model
                output_cgm = student_model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
                
                # Calculate loss against the TRUE target
                val_loss = criterion(student_model.normalize_cgm(output_cgm), student_model.normalize_cgm(target_cgm))
                val_losses.append(val_loss.item())

        # Debug Print 4: Confirm all validation batches processed
        print("\nFinished processing all validation batches.") # Use '\n' to move to next line after the '\r' prints

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Iter {iteration} - STUDENT Validation CGM Loss: {avg_val_loss:.6f}")

        # Save the STUDENT model checkpoint
        # Debug Print 5: Indicate attempting to save the model
        print(f"Attempting to save model at iteration {iteration}...")
        model_filename = f"{save_path}student_model_iter_{iteration}.pth"
        # If the hang occurs here, it's during the torch.save operation
        torch.save(student_model.state_dict(), model_filename)
        # Debug Print 6: Confirm model saved
        print(f"Model saved: {model_filename}")

        student_model.train() # Set the student back to training mode for the next loop
        # Debug Print 7: Confirm student model is back in train mode
        print("Student model set back to train mode.")
        # Debug Print 8: Indicate the end of the validation block
        print("--- Validation Block Finished ---")

# --- End of the while loop ---

print("Training and Validation complete!")
