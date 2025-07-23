import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CGMPredictor
from dataset import CGMDataset, collate_fn
from collections import OrderedDict

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "results/"
TEST_DATA_PATH = "./Datasets/Test" # The final hold-out test set
BATCH_SIZE = 2 # Can be larger for evaluation
# This must match the sequence length the models were trained on
MAX_LEN = 512 * 10 

print(f"Using device: {DEVICE}")

# --- Load Final Test Data ---
print(f"Loading final test dataset from: {TEST_DATA_PATH}")
try:
    test_dataset = CGMDataset(file=TEST_DATA_PATH, max_len=MAX_LEN, pred_time=90, augment=False, max_range=1.5)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )
    print("Test dataset loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Test data not found at '{TEST_DATA_PATH}'. Please make sure the directory exists.")
    exit()

# --- Find All Model Checkpoints ---
all_checkpoints = [f for f in os.listdir(MODEL_DIR) if (f.startswith('student_model_iter_') or f.startswith('model_iter_')) and f.endswith('.pth')]

if not all_checkpoints:
    print(f"No model checkpoints found in the '{MODEL_DIR}' directory.")
    exit()

print(f"\nFound {len(all_checkpoints)} model checkpoints to evaluate.")

# --- Evaluation Loop ---
all_results = {}
criterion = nn.L1Loss()

for model_name in all_checkpoints:
    checkpoint_path = os.path.join(MODEL_DIR, model_name)
    print(f"\n--- Evaluating: {model_name} ---")

    is_student = model_name.startswith('student_model_')
    
    # 1. Create the correct model architecture
    if is_student:
        print("-> Type: Student Model")
        model = CGMPredictor(n_embd=4*48, n_head=4, n_layer=2)
    else: # It's a teacher model
        print("-> Type: Teacher Model")
        model = CGMPredictor(n_embd=8*48, n_head=8, n_layer=3)

    # 2. Load the state dictionary with the correct logic for each type
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        # The original teacher models were compiled, their keys need cleaning.
        if not is_student:
            print("-> Applying key cleaning for compiled teacher model...")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    name = k[10:] # remove `_orig_mod.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            # Student models were NOT compiled and can be loaded directly
            model.load_state_dict(state_dict)
            
    except Exception as e:
        print(f"--> ERROR: Failed to load model {model_name}. Reason: {e}")
        continue # Skip to the next model

    model.to(DEVICE)
    model.eval()

    # 3. Run evaluation on the test set
    test_losses = []
    with torch.no_grad():
        for test_batch in test_loader:
            # Move batch to device...
            cgm = test_batch['cgm'].to(DEVICE); basal = test_batch['basal'].to(DEVICE); bolus = test_batch['bolus'].to(DEVICE)
            cgm_time = test_batch['cgm_time'].to(DEVICE); basal_time = test_batch['basal_time'].to(DEVICE); bolus_time = test_batch['bolus_time'].to(DEVICE)
            target_cgm = test_batch['target_cgm'].to(DEVICE); target_time = test_batch['target_time'].to(DEVICE); pred_time = test_batch['pred_time'].to(DEVICE)

            output_cgm = model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
            test_loss = criterion(model.normalize_cgm(output_cgm), model.normalize_cgm(target_cgm))
            test_losses.append(test_loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    all_results[model_name] = avg_test_loss
    print(f"--> Result: Test Loss = {avg_test_loss:.6f}")

# --- Final Report ---
print("\n" + "="*30)
print("--- FINAL TEST REPORT ---")
print("="*30)

if all_results:
    # Separate results into student and teacher groups
    student_results = {k: v for k, v in all_results.items() if k.startswith('student_model_')}
    teacher_results = {k: v for k, v in all_results.items() if k.startswith('model_iter_')}

    # 1. Announce Best Student Model
    if student_results:
        best_student_name = min(student_results, key=student_results.get)
        best_student_loss = student_results[best_student_name]
        print(f"\n✅ Best Student Model: {best_student_name}")
        print(f"   Test Loss: {best_student_loss:.6f}")
    else:
        print("\nNo student models were evaluated.")

    # 2. Announce Best Teacher Model
    if teacher_results:
        best_teacher_name = min(teacher_results, key=teacher_results.get)
        best_teacher_loss = teacher_results[best_teacher_name]
        print(f"\n✅ Best Teacher Model: {best_teacher_name}")
        print(f"   Test Loss: {best_teacher_loss:.6f}")
    else:
        print("\nNo teacher models were evaluated.")

    # 3. Announce Overall Winner
    best_overall_name = min(all_results, key=all_results.get)
    best_overall_loss = all_results[best_overall_name]
    
    print("\n" + "#"*40)
    print(f"🏆 OVERALL BEST PERFORMING MODEL 🏆")
    print(f"   Model: {best_overall_name}")
    print(f"   Test Loss: {best_overall_loss:.6f}")
    print("#"*40)
    
else:
    print("No results were generated.")