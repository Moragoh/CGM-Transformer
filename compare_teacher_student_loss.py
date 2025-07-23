import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CGMPredictor
from dataset import CGMDataset, collate_fn
from collections import OrderedDict

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "results/"
TEST_DATA_PATH = "./Datasets/Test"
BATCH_SIZE = 2 # Reduced to prevent OutOfMemory errors
MAX_LEN = 512 * 10 
LOG_FILE_PATH = os.path.join(MODEL_DIR, "final_evaluation_report.txt")

# --- Set up Logging to both Console and File ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect print statements to our custom logger
sys.stdout = Logger(LOG_FILE_PATH)

print(f"Using device: {DEVICE}")
print(f"Logging results to: {LOG_FILE_PATH}")

# --- Load Final Test Data ---
print(f"\nLoading final test dataset from: {TEST_DATA_PATH}")
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
    sys.exit()

# --- Find All Model Checkpoints ---
all_checkpoints = [f for f in os.listdir(MODEL_DIR) if (f.startswith('student_model_iter_') or f.startswith('model_iter_')) and f.endswith('.pth')]

if not all_checkpoints:
    print(f"No model checkpoints found in the '{MODEL_DIR}' directory.")
    sys.exit()

print(f"\nFound {len(all_checkpoints)} model checkpoints to evaluate.")

# --- Evaluation Loop ---
all_results = {}
criterion = nn.L1Loss()

for model_name in sorted(all_checkpoints): # Sort for consistent order
    checkpoint_path = os.path.join(MODEL_DIR, model_name)
    print(f"\n--- Evaluating: {model_name} ---")

    is_student = model_name.startswith('student_model_')
    
    # Create the correct model architecture
    if is_student:
        print("-> Type: Student Model")
        model = CGMPredictor(n_embd=4*48, n_head=4, n_layer=2)
    else:
        print("-> Type: Teacher Model")
        model = CGMPredictor(n_embd=8*48, n_head=8, n_layer=3)

    # Load the state dictionary
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        if not is_student:
            print("-> Applying key cleaning for compiled teacher model...")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    name = k[10:]
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
    except Exception as e:
        print(f"--> ERROR: Failed to load model {model_name}. Reason: {e}")
        continue

    model.to(DEVICE)
    model.eval()

    # Run evaluation on the test set
    test_losses = []
    with torch.no_grad():
        for test_batch in test_loader:
            cgm = test_batch['cgm'].to(DEVICE); basal = test_batch['basal'].to(DEVICE); bolus = test_batch['bolus'].to(DEVICE)
            cgm_time = test_batch['cgm_time'].to(DEVICE); basal_time = test_batch['basal_time'].to(DEVICE); bolus_time = test_batch['bolus_time'].to(DEVICE)
            target_cgm = test_batch['target_cgm'].to(DEVICE); target_time = test_batch['target_time'].to(DEVICE); pred_time = test_batch['pred_time'].to(DEVICE)

            output_cgm = model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
            test_loss = criterion(model.normalize_cgm(output_cgm), model.normalize_cgm(target_cgm))
            test_losses.append(test_loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')
    all_results[model_name] = avg_test_loss
    print(f"--> Result: Test Loss = {avg_test_loss:.6f}")

# --- Final Report ---
print("\n" + "="*40)
print("--- FINAL TEST PERFORMANCE REPORT ---")
print("="*40)

if all_results:
    student_results = {k: v for k, v in all_results.items() if k.startswith('student_model_')}
    teacher_results = {k: v for k, v in all_results.items() if k.startswith('model_iter_')}

    if student_results:
        best_student_name = min(student_results, key=student_results.get)
        print(f"\n✅ Best Student Model: {best_student_name} (Loss: {student_results[best_student_name]:.6f})")
    
    if teacher_results:
        best_teacher_name = min(teacher_results, key=teacher_results.get)
        print(f"✅ Best Teacher Model: {best_teacher_name} (Loss: {teacher_results[best_teacher_name]:.6f})")

    # Overall Winner and Ranking
    sorted_results = sorted(all_results.items(), key=lambda item: item[1])
    best_overall_name, best_overall_loss = sorted_results[0]
    
    print("\n" + "#"*40)
    print(f"🏆 OVERALL BEST PERFORMING MODEL 🏆")
    print(f"   Model: {best_overall_name}")
    print(f"   Test Loss: {best_overall_loss:.6f}")
    print("#"*40)

    print("\nFull Performance Ranking (lower is better):")
    for i, (model_name, loss) in enumerate(sorted_results):
        print(f"{i+1: >2}. {model_name:<35} Test Loss: {loss:.6f}")
else:
    print("No results were generated.")