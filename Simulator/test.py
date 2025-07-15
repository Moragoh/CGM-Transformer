from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

from model import CGMPredictor
import pandas as pd
import torch
import numpy as np



max_len = 512*10

model = CGMPredictor(
    n_embd=384,
    n_head=8,
    n_layer=3,
    dropout=0.3
)


model = torch.compile(model)
print(f'Trainable parameters: {model.num_params()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


model.load_state_dict(torch.load('./model_iter_19000.pth', map_location=torch.device(device)))

class MyController(Controller):
    def __init__(self, P=1, I=0, D=0, target=90):

        self.cgm_his = []
        self.cgm_time = []
        self.basal_his = [0]
        self.basal_time = [0]
        self.bolus_his = []
        self.bolus_time = []

        self.base_rate = 0.013
        self.ins_dur = 280

        #adolescent 1: 0.025
        #adolescent 2: 0.025
        #adult 4: 0.015


        self.P = 0.012
        self.I = 0#.00000010
        self.D = 0#.0006
        self.target = target
        self.integrated_state = 0

    def add_cgm_reading(self, cgm):
        self.cgm_time.append(len(self.cgm_his) * 2/3)
        self.cgm_his.append(cgm)


    def predict(self, t):
        cgm = torch.Tensor(self.cgm_his).to(device)
        bolus = torch.Tensor(self.bolus_his).to(device)
        cgm_time = torch.Tensor(self.cgm_time).to(device)
        bolus_time = torch.Tensor(self.bolus_time).to(device)
        target_time = torch.Tensor(t).to(device)
        pred_time = torch.Tensor([0]*len(t)).to(device)

        
        basal_time = self.basal_time + [self.basal_time[-1] + 1, max(t) - 0.5]
        basal_his = self.basal_his + [0, 0]
        
        basal_time = torch.Tensor(basal_time).to(device)
        basal = torch.Tensor(basal_his).to(device)
        
        
        cgm = cgm[-max_len:]
        cgm_time = cgm_time[-max_len:]

        basal = basal[basal_time >= cgm_time[0]] * 60
        basal_time = basal_time[basal_time >= cgm_time[0]]
        bolus = bolus[bolus_time >= cgm_time[0]] * 60
        bolus_time = bolus_time[bolus_time >= cgm_time[0]]

        
        return model(
            cgm.unsqueeze(0), 
            basal.unsqueeze(0), 
            bolus.unsqueeze(0), 
            (cgm_time - cgm_time[0] + 2/3).unsqueeze(0), 
            (basal_time - cgm_time[0] + 2/3).unsqueeze(0), 
            (bolus_time - cgm_time[0] + 2/3).unsqueeze(0), 
            (target_time - cgm_time[0] + 2/3).unsqueeze(0), 
            pred_time.unsqueeze(0)
        )[0]

    def calculate_iob(self):
        recent = torch.tensor(self.basal_his[-self.ins_dur:], dtype=torch.float32)
        bolus_vals = torch.tensor(self.bolus_his, dtype=torch.float32)
        bolus_times = torch.tensor(self.bolus_time, dtype=torch.float32)
        cutoff_time = self.cgm_time[-self.ins_dur:][0]
        mask = bolus_times > cutoff_time
    
        filtered_bolus = bolus_vals[mask]
        filtered_bolus_times = bolus_times[mask]
    
        time_indices = ((filtered_bolus_times - cutoff_time) / (2/3)).int()
        time_indices = torch.clamp(time_indices, 0, self.ins_dur - 1)
        
        for idx, bolus in zip(time_indices, filtered_bolus):
            recent[idx] += bolus
            
        weights = torch.arange(self.ins_dur + 1 - len(recent), self.ins_dur + 1, dtype=torch.float32) / self.ins_dur
        ins_ob = torch.sum(recent * weights)
        return ins_ob.item() + max(self.ins_dur - len(self.cgm_time), 0) * self.base_rate

        
    def policy(self, observation, reward, done, **kwargs):

        cgm = observation.CGM
        pname = kwargs.get('patient_name')

        if 'adult' in pname:
            self.ins_dur = 40
            self.base_rate = 0.013
            self.P = 0.012
        elif 'adolescent#001' in pname:
            self.ins_dur = 280
            self.base_rate = 0.025
            self.P = 0.03
        elif 'adolescent#002' in pname:
            self.ins_dur = 280
            self.base_rate = 0.025
            self.P = 0.03
        elif 'adolescent#003' in pname:
            self.ins_dur = 40
            self.base_rate = 0.013
            self.P = 0.003
        elif 'adolescent#004' in pname:
            self.ins_dur = 280
            self.base_rate = 0.003
            self.P = 0.025
        elif 'adolescent#005' in pname:
            self.ins_dur = 40
            self.base_rate = 0.013
            self.P = 0.006
        elif 'adolescent#006' in pname:
            self.ins_dur = 40
            self.base_rate = 0.015
            self.P = 0.01
        elif 'adolescent#007' in pname:
            self.ins_dur = 200
            self.base_rate = 0.008
            self.P = 0.019
        elif 'adolescent#008' in pname:
            self.ins_dur = 200
            self.base_rate = 0.006
            self.P = 0.026
        elif 'adolescent#009' in pname:
            self.ins_dur = 40
            self.base_rate = 0.008
            self.P = 0.006
        elif 'adolescent#010' in pname:
            self.ins_dur = 40
            self.base_rate = 0.008
            self.P = 0.008
        else:
            self.base_rate = 0.013
            self.P = 0.008
        
        self.add_cgm_reading(cgm)


        
        bg = self.predict([self.cgm_time[-1] + 12]).item()
        

        ins_target = self.P * (bg - self.target)
        ins_ob = self.calculate_iob()
        control_input = ins_target - ins_ob

        if bg < 70: 
            control_input = 0
        else:
            control_input = max(control_input, self.base_rate)


        self.basal_his.append(control_input)
        self.basal_time.append(self.cgm_time[-1])
        
        print(f"{len(self.cgm_time)} | Cur: {int(cgm)} | Prediction: {int(bg)} | {control_input}")
        
        self.basal_his[-1] = control_input
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):

        self.cgm_his = []
        self.cgm_time = []
        self.basal_his = [0]
        self.basal_time = [0]
        self.bolus_his = []
        self.bolus_time = []



ctrller = MyController(0)
simulate(controller=ctrller)