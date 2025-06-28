from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
import torch._dynamo
torch._dynamo.config.suppress_errors = True

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
    def __init__(self, P=1, I=0, D=0, target=100):

        self.cgm_his = []
        self.cgm_time = []
        self.basal_his = [0]
        self.basal_time = [0]
        self.bolus_his = []
        self.bolus_time = []

        self.base_rate = 0.02

        #adolescent 1: 0.025
        #adolescent 2: 0.025
        #adult 4: 0.015


        self.P = 0.013
        self.I = 0#.00000010
        self.D = 0#.0006
        self.target = target
        self.integrated_state = 0

    def add_cgm_reading(self, cgm):
        self.cgm_time.append(len(self.cgm_his) * 2/3)
        self.cgm_his.append(cgm)


    def predict(self, t):
        cgm = torch.Tensor(self.cgm_his)
        basal = torch.Tensor(self.basal_his)
        bolus = torch.Tensor(self.bolus_his)
        cgm_time = torch.Tensor(self.cgm_time)
        basal_time = torch.Tensor(self.basal_time)
        bolus_time = torch.Tensor(self.bolus_time)
        target_time = torch.Tensor(t)
        pred_time = torch.Tensor([self.base_rate]*len(t))
        
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

    def policy(self, observation, reward, done, **kwargs):
   
        #if len(self.cgm_his) < 20 * 24:
        #    self.P = 0.0005
        #else:
        #    self.P = 0.001

        cgm = observation.CGM
        self.add_cgm_reading(cgm)
    
        #last_time = self.cgm_time[-1]
        #target = [last_time + i for i in range(1,6)]
        #res = self.predict(target)
        
        #self.cgm_his.extend(res)
        #self.cgm_time.extend(target)
        #self.basal_time.extend(target)
        #self.basal_his.extend([0]*len(target))
        
        bg = self.predict([self.cgm_time[-1] + 12]).item()
        
        #self.cgm_his = self.cgm_his[:-len(target)]
        #self.cgm_time = self.cgm_time[:-len(target)]
        #self.basal_his = self.basal_his[:-len(target)]
        #self.basal_time = self.basal_time[:-len(target)]


        ins_target = self.P * (bg - self.target)
        recent = torch.Tensor(self.basal_his[-120:])
        ins_ob = sum(recent * torch.arange(121-len(recent),121)/120)

        ################################
        ################################

        
        control_input = ins_target - ins_ob

        print(f"Cur: {int(cgm)} | Prediction: {int(bg)} | {control_input}")

        if bg < 100:
            control_input = 0
        else:
            #control_input = min(6-sum(self.basal_his[-90:]), control_input)  
            control_input = max(control_input, self.base_rate)


        self.basal_his.append(control_input)
        self.basal_time.append(self.cgm_time[-1])
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):
        self.integrated_state = 0



    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''

        self.cgm_his = []
        self.cgm_time = []
        self.basal_his = [0]
        self.basal_time = [0]
        self.bolus_his = []
        self.bolus_time = []



ctrller = MyController(0)
simulate(controller=ctrller)