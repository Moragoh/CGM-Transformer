import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CGMPredictor
from dataset import CGMDataset, collate_fn
from collections import OrderedDict
from tqdm import tqdm
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "results/"
TEST_DATA_PATH = "./Datasets/Test"
TEACHER_BENCHMARK_FILE = "model_iter_23000.pth"
BATCH_SIZE = 2
MAX_LEN = 512 * 10
LOG_FILE_PATH = os.path.join(MODEL_DIR, "advanced_comparison_report.txt")

# --- Set up Logging ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message); self.log.write(message)
    def flush(self):
        self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE_PATH)

# --- Helper Function to Evaluate a Single Model (Overall Performance) ---
def get_overall_performance(model_name, model_dir, test_loader):
    print(f"\n--- Getting Overall Performance for: {model_name} ---")
    is_student = model_name.startswith('student_model_')
    model_path = os.path.join(model_dir, model_name)
    
    if is_student:
        model = CGMPredictor(n_embd=4*48, n_head=4, n_layer=2)
    else:
        model = CGMPredictor(n_embd=8*48, n_head=8, n_layer=3)
        
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = OrderedDict([(k[10:], v) for k, v in state_dict.items()])
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"--> ERROR loading {model_name}: {e}. Skipping.")
        return None, None
        
    model.to(DEVICE); model.eval()
    
    total_abs_error, total_squared_error, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            cgm, basal, bolus = batch['cgm'].to(DEVICE), batch['basal'].to(DEVICE), batch['bolus'].to(DEVICE)
            cgm_time, basal_time, bolus_time = batch['cgm_time'].to(DEVICE), batch['basal_time'].to(DEVICE), batch['bolus_time'].to(DEVICE)
            target_cgm, target_time, pred_time = batch['target_cgm'].to(DEVICE), batch['target_time'].to(DEVICE), batch['pred_time'].to(DEVICE)
            
            output_cgm = model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
            error = output_cgm - target_cgm
            
            total_abs_error += torch.sum(torch.abs(error)).item()
            total_squared_error += torch.sum(error**2).item()
            total_samples += error.numel()

    mae = total_abs_error / total_samples if total_samples > 0 else float('inf')
    rmse = (total_squared_error / total_samples)**0.5 if total_samples > 0 else float('inf')
    print(f"--> Overall MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

# --- Helper Function for Detailed Interval Analysis ---
def get_interval_performance(model_name, model_dir, test_loader):
    print(f"\n--- Getting Interval Performance for: {model_name} ---")
    is_student = model_name.startswith('student_model_')
    model_path = os.path.join(model_dir, model_name)
    
    if is_student:
        model = CGMPredictor(n_embd=4*48, n_head=4, n_layer=2)
    else:
        model = CGMPredictor(n_embd=8*48, n_head=8, n_layer=3)
        
    state_dict = torch.load(model_path, map_location=DEVICE)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = OrderedDict([(k[10:], v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict)
    model.to(DEVICE); model.eval()

    num_intervals = 18
    total_abs_error = torch.zeros(num_intervals, device=DEVICE)
    total_squared_error = torch.zeros(num_intervals, device=DEVICE)
    total_samples = torch.zeros(num_intervals, device=DEVICE)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Analyzing {model_name}"):
            cgm, basal, bolus = batch['cgm'].to(DEVICE), batch['basal'].to(DEVICE), batch['bolus'].to(DEVICE)
            cgm_time, basal_time, bolus_time = batch['cgm_time'].to(DEVICE), batch['basal_time'].to(DEVICE), batch['bolus_time'].to(DEVICE)
            target_cgm, target_time, pred_time = batch['target_cgm'].to(DEVICE), batch['target_time'].to(DEVICE), batch['pred_time'].to(DEVICE)
            
            output_cgm = model(cgm, basal, bolus, cgm_time, basal_time, bolus_time, target_time, pred_time)
            error = output_cgm - target_cgm
            
            for i in range(1, num_intervals + 1):
                mask = (pred_time == i)
                if mask.any():
                    total_abs_error[i-1] += torch.sum(torch.abs(error[mask]))
                    total_squared_error[i-1] += torch.sum(error[mask]**2)
                    total_samples[i-1] += mask.sum()
    
    # Avoid division by zero
    total_samples[total_samples == 0] = 1.0
    mae_by_interval = (total_abs_error / total_samples).cpu()
    rmse_by_interval = torch.sqrt(total_squared_error / total_samples).cpu()
    
    return mae_by_interval, rmse_by_interval

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load Test Data
    print(f"\nLoading test dataset from: {TEST_DATA_PATH}...")
    try:
        test_dataset = CGMDataset(file=TEST_DATA_PATH, max_len=MAX_LEN, pred_time=90, augment=False, max_range=1.5)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=False)
    except FileNotFoundError:
        print(f"ERROR: Test data not found at '{TEST_DATA_PATH}'. Exiting."); sys.exit(1)

    # --- PART 1: Find the Best Student Model ---
    print("\n" + "="*55)
    print("--- PART 1: IDENTIFYING THE BEST STUDENT MODEL ---")
    print("="*55)
    student_checkpoints = [f for f in os.listdir(MODEL_DIR) if f.startswith('student_model_')]
    if not student_checkpoints:
        print("No student models found. Exiting."); sys.exit(1)
        
    student_performances = {}
    for student_name in student_checkpoints:
        mae, _ = get_overall_performance(student_name, MODEL_DIR, test_loader)
        if mae is not None:
            student_performances[student_name] = mae
    
    if not student_performances:
        print("Could not evaluate any student models. Exiting."); sys.exit(1)

    best_student_name = min(student_performances, key=student_performances.get)
    print(f"\n🏆 Best Student Model identified (by overall MAE): {best_student_name}")

    # --- PART 2: Detailed Interval Comparison ---
    print("\n" + "="*55)
    print("--- PART 2: DETAILED INTERVAL PERFORMANCE ANALYSIS ---")
    print("="*55)
    
    # Analyze the benchmark teacher model
    teacher_mae, teacher_rmse = get_interval_performance(TEACHER_BENCHMARK_FILE, MODEL_DIR, test_loader)
    
    # Analyze the champion student model
    student_mae, student_rmse = get_interval_performance(best_student_name, MODEL_DIR, test_loader)
    
    # --- FINAL REPORT ---
    print("\n" + "#"*75)
    print("--- FINAL REPORT: TEACHER VS. BEST STUDENT (PERFORMANCE BY INTERVAL) ---")
    print("#"*75)
    print(f"Teacher Model: {TEACHER_BENCHMARK_FILE}")
    print(f"Student Model: {best_student_name}")
    print("\n" + "-"*75)
    header = f"{'Time (min)':<12} | {'Teacher MAE':<15} | {'Student MAE':<15} | {'Teacher RMSE':<15} | {'Student RMSE':<15}"
    print(header)
    print("-"*75)
    
    pred_times_in_minutes = (1 + torch.arange(18)) * 5
    for i in range(18):
        time = pred_times_in_minutes[i].item()
        t_mae, t_rmse = teacher_mae[i].item(), teacher_rmse[i].item()
        s_mae, s_rmse = student_mae[i].item(), student_rmse[i].item()
        print(f"{time:<12} | {t_mae:<15.4f} | {s_mae:<15.4f} | {t_rmse:<15.4f} | {s_rmse:<15.4f}")
    print("-"*75)