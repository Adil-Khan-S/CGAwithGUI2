from tkinter.filedialog import *
from tkinter import *
from tkinter import ttk
import os
import pandas as pd
from mat4py import loadmat
import functions_gait_phases_29_11_2022 as functions_new
from joblib import dump, load
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

top = Tk()

top.geometry("500x350")
top.title("Post Gait CGA Simulator")

global treatment

def show():
    global patFolder
    patFolder = askdirectory()

def show1():
    treatmentL = [0,0,0,0,0]
    treatmentR = [0,0,0,0,0]
    if (muscle0Cmb.get() == "Yes"):
        treatmentL[0] = 1
    if (muscle1Cmb.get() == "Yes"):
        treatmentL[1] = 1
    if (muscle2Cmb.get() == "Yes"):
        treatmentL[2] = 1
    if (muscle3Cmb.get() == "Yes"):
        treatmentL[3] = 1
    if (muscle4Cmb.get() == "Yes"):
        treatmentL[4] = 1

    if (muscle0RCmb.get() == "Yes"):
        treatmentR[0] = 1
    if (muscle1RCmb.get() == "Yes"):
        treatmentR[1] = 1
    if (muscle2RCmb.get() == "Yes"):
        treatmentR[2] = 1
    if (muscle3RCmb.get() == "Yes"):
        treatmentR[3] = 1
    if (muscle4RCmb.get() == "Yes"):
        treatmentR[4] = 1

    readMatFilesOfPatients(treatmentL,treatmentR)

def readMatFilesOfPatients(treatmentL,treatmentR):
    load_files()
    pat_name = 1
    pre_treat_df_all_cycles = pd.DataFrame()
    print('Folder Name : ', patFolder)
    pre_path = patFolder + "\\PRE"
    files = os.listdir(pre_path)
    total_pre_cycles = 1
    for file in files:
        path = patFolder + "\\PRE\\" + file
        print('File : ', path)
        mat = loadmat(path)
        EA_Filter = mat['EAFilter']
        Events = mat['Events']
        freq = EA_Filter['LKnee']['Rate']
        right_cycle, right_cycle_stance, right_cycle_swing, left_cycle, left_cycle_stance, left_cycle_swing = functions_new.get_ankle_knee_angles(
            EA_Filter, Events, rate=freq)

        left_interp_cycle, left_interp_stance_cycle, left_interp_swing_cycle, right_interp_cycle, right_interp_stance_cycle, right_interp_swing_cycle, st_prop, sw_prop = functions_new.cycle_normalization(
            Events, right_cycle, right_cycle_stance, right_cycle_swing, left_cycle, left_cycle_stance, left_cycle_swing,
            rate=freq)

        for i in range(len(left_interp_cycle)):
            # Left Side Stance Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Left"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Stance"

            for index, value in enumerate(left_interp_stance_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value

            my_dict["t1"] = treatmentL[0]
            my_dict["t2"] = treatmentL[1]
            my_dict["t3"] = treatmentL[2]
            my_dict["t4"] = treatmentL[3]
            my_dict["t5"] = treatmentL[4]

            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            # Left Side Swing Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Left"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Swing"

            for index, value in enumerate(left_interp_swing_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value

            my_dict["t1"] = treatmentL[0]
            my_dict["t2"] = treatmentL[1]
            my_dict["t3"] = treatmentL[2]
            my_dict["t4"] = treatmentL[3]
            my_dict["t5"] = treatmentL[4]

            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            # Right Side
            print('Right side shape after interpolation : ', right_interp_cycle[i].shape)
            # Right Side Stance Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Right"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Stance"

            for index, value in enumerate(right_interp_stance_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value

            my_dict["t1"] = treatmentR[0]
            my_dict["t2"] = treatmentR[1]
            my_dict["t3"] = treatmentR[2]
            my_dict["t4"] = treatmentR[3]
            my_dict["t5"] = treatmentR[4]

            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            # Right Side Swing Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Right"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Swing"

            for index, value in enumerate(right_interp_swing_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value

            my_dict["t1"] = treatmentR[0]
            my_dict["t2"] = treatmentR[1]
            my_dict["t3"] = treatmentR[2]
            my_dict["t4"] = treatmentR[3]
            my_dict["t5"] = treatmentR[4]

            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            total_pre_cycles = total_pre_cycles + 1

        print(pre_treat_df_all_cycles)
        df_test_data = pre_treat_df_all_cycles
        #print(df_test_data.iloc[0,4:10])
        #break
        # saving the dataframe
        #pre_treat_df_all_cycles.to_csv(patFolder+'\\patient1.csv')

        df_test_data_stance = df_test_data[df_test_data['Phase'] == "Stance"]
        df_test_data_swing = df_test_data[df_test_data['Phase'] == "Swing"]

        #df_test_data_stance = df_test_data_stance.drop("Unnamed: 0", axis='columns')
        #df_test_data_swing = df_test_data_swing.drop("Unnamed: 0", axis='columns')
        #df_test_data_stance["index"] = df_test_data_stance.index
        #df_test_data_stance.reset_index()
        #df_test_data_swing["index"] = df_test_data_swing.index
        #df_test_data_swing.reset_index()

        ###print(df_test_data_stance)
        ###print(df_test_data_swing)

        data_pre_test_stance = []
        data_treatment_test_stance = []
        for ind in df_test_data_stance.index:
            #print(ind)
            #print(df_test_data_stance.loc[ind])
            #break
            tmp = df_test_data_stance.loc[ind]
            pre = tmp[4:106]
            treatment = tmp[106: 111].copy().values.tolist()
            #print(pre)
            #break
            pre_1 = []
            pre_2 = []

            for j in range(1, 52):
                pre_1.append(pre[j])
            for j in range(52, 103):
                pre_2.append(pre[j])

            t = treatment

            if 1 in t:
                data_pre_test_stance.append([pre_1, pre_2])
                data_treatment_test_stance.append(t)

        data_pre_test_stance = np.array(data_pre_test_stance)
        data_pre_test_stance = scaler_stance_pre.fit_transform(
            data_pre_test_stance.reshape(-1, data_pre_test_stance.shape[-1])).reshape(data_pre_test_stance.shape)

        data_pre_test_swing = []
        data_treatment_test_swing = []

        for ind in df_test_data_swing.index:

            tmp = df_test_data_swing.loc[ind]
            pre = tmp[4:106]
            treatment = tmp[106: 111].copy().values.tolist()

            pre_1 = []
            pre_2 = []

            for j in range(1, 52):
                pre_1.append(pre[j])
            for j in range(52, 103):
                pre_2.append(pre[j])

            t = treatment

            if 1 in t:
                data_pre_test_swing.append([pre_1, pre_2])
                data_treatment_test_swing.append(t)

    data_pre_test_swing = np.array(data_pre_test_swing)
    data_pre_test_swing = scaler_swing_pre.fit_transform(
            data_pre_test_swing.reshape(-1, data_pre_test_swing.shape[-1])).reshape(data_pre_test_swing.shape)

    test_set_stance = GateTestDataset(np.array(data_pre_test_stance), np.array(data_treatment_test_stance))
    test_loader_stance = torch.utils.data.DataLoader(test_set_stance, batch_size=16, shuffle=True)

    test_set_swing = GateTestDataset(np.array(data_pre_test_swing), np.array(data_treatment_test_swing))
    test_loader_swing = torch.utils.data.DataLoader(test_set_swing, batch_size=16, shuffle=True)

    output_stance = test_model(stance_model, test_loader_stance, "stance")
    output_swing = test_model(swing_model, test_loader_swing, "swing")

    plt.figure(figsize=(8, 5))
    plt.title("Knee Joint")
    df_test_data.iloc[0, 4:10]
    kwargs = {'color': 'green'}
    kwargs2 = {'color': 'red'}
    for i in range(0,len(df_test_data),2):
        plt.plot(merge_st_sw_phase(df_test_data.iloc[i, 4:55], df_test_data.iloc[i+1, 4:55], st_prop, sw_prop),color='green')
    plt.plot([], [], label='Pre CGA', **kwargs)

    for i in range(len(output_stance)):
        plt.plot(merge_st_sw_phase(output_stance[i, 0], output_swing[i, 0], st_prop, sw_prop),
                 color='red', linestyle='dashed')
    plt.plot([], [], label='Predicted Post CGA (mean of all cycles)', **kwargs)

    #plt.plot(merge_st_sw_phase(np.mean(output_stance[:, 0], axis=0), np.mean(output_swing[:, 0], axis=0), st_prop, sw_prop),color='red',label='Predicted Post CGA (mean of all cycles)', linestyle='dashed')
    plt.ylabel("Knee Flexion (°)")
    plt.xlabel("Cycle Instant (%)")
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("Ankle Joint")
    df_test_data.iloc[0, 4:10]
    kwargs = {'color': 'green'}
    for i in range(0,len(df_test_data),2):
        plt.plot(merge_st_sw_phase(df_test_data.iloc[i, 55:106], df_test_data.iloc[i+1, 55:106], st_prop, sw_prop),color='green')
    plt.plot([], [], label='Pre CGA', **kwargs)
    plt.plot(merge_st_sw_phase(np.mean(output_stance[:, 1], axis=0), np.mean(output_swing[:, 1], axis=0), st_prop, sw_prop),color='red',label='Predicted Post CGA (mean of all cycles)', linestyle='dashed')
    plt.ylabel("Knee Flexion (°)")
    plt.xlabel("Cycle Instant (%)")
    plt.legend(loc='best')
    plt.show()

MAIN_PATH = "D:/DataForCGAGUI/"
def load_files():
    global scaler_stance_pre, scaler_swing_pre, stance_model, swing_model, scaler_stance_post, scaler_swing_post
    scaler_stance_pre = load(MAIN_PATH + 'scaler_stance_pre.bin')
    scaler_swing_pre = load(MAIN_PATH + 'scaler_swing_pre.bin')
    scaler_stance_post = load(MAIN_PATH + 'scaler_stance_post.bin')
    scaler_swing_post = load(MAIN_PATH + 'scaler_swing_post.bin')

    stance_model = torch.load(MAIN_PATH + 'biLSTM_treatment_woMTD_attention.pt',map_location=torch.device('cpu'))
    stance_model.eval()

    swing_model = torch.load('D:/DataForCGAGUI/swing_biLSTM_treatment_woMTD_attention.pt',map_location=torch.device('cpu'))
    swing_model.eval()

def create_bi_h(h, gate, batch_size):
    #tmp = torch.Tensor([]).cuda()
    tmp = torch.Tensor([])
    for t in gate.mT:
        tmp_ = torch.ones((2, batch_size, 51))
        #tmp_ = torch.ones((2, batch_size, 51)).cuda()
        for t_1, t_2 in zip(tmp_[0], t):
            t_1 = t_1 * t_2
        for t_1, t_2 in zip(tmp_[1], t):
            t_1 = t_1 * t_2

        tmp = torch.cat((tmp, tmp_.unsqueeze(0)), 0)

    return tmp

class biLSTM_treatment_woMTD_attention(nn.Module):

    def __init__(self, input_size, output_size):
        super(biLSTM_treatment_woMTD_attention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ls = []
        for i in range(5):
            lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.output_size,
                            batch_first = True,
                            bidirectional  = True)
            self.ls.append(lstm)
            #self.ls.append(lstm.cuda())
        self.attention = nn.MultiheadAttention(len(self.ls) * 51 * 2, num_heads=1)
        self.l1 = nn.Linear(len(self.ls) * 51 * 2, 124)
        self.l2 = nn.Linear(124, 51)


    def forward(self, x, gate):
        output = torch.Tensor([])
        #output = torch.Tensor([]).cuda()
        batch_size = x.shape[0]
        h0 = torch.ones(2, batch_size, self.output_size).requires_grad_()
        tmp = create_bi_h(h0, gate, batch_size)
        #h0 = torch.ones(2, batch_size, self.output_size).requires_grad_().cuda()
        #tmp = create_bi_h(h0, gate, batch_size).cuda()
        for i in range(len(self.ls)):
            c0 = torch.zeros(2, batch_size, self.output_size).requires_grad_()
            v = self.ls[i]
            #c0 = torch.zeros(2, batch_size, self.output_size).requires_grad_().cuda()
            #v = self.ls[i].cuda()
            v = v(x, (c0, c0))
            output = torch.cat((output, v[0].unsqueeze(1)), 1).requires_grad_()
            #output = torch.cat((output, v[0].unsqueeze(1)), 1).requires_grad_().cuda()

        # output = self.dropout(output)

        output = output.view(batch_size, 2, 510)
        lstm_output = output
        lstm_output = lstm_output.permute(1, 0, 2)
        #print(lstm_output.size())
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        attention_output = attention_output.permute(1, 0, 2)
        # output = self.dropout(output)
        output = attention_output
        output = F.relu(self.l1(output))
        output = self.l2(output)
        return output

class GateTestDataset(torch.utils.data.Dataset):

  def __init__(self, x, gate):
    if isinstance(x, pd.DataFrame):
      x = x.values
      gate = gate.values

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.gate_train=torch.tensor(gate,dtype=torch.float32)

  def __len__(self):
    return len(self.x_train)

  def __getitem__(self,idx):

    return self.x_train[idx], self.gate_train[idx]

def test_model(net, test_loader, phase):

        output_val = np.array([])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        val_loss = 0.0

        with torch.no_grad():
            net.eval()
            for data in test_loader:
                inputs, gate = data
                inputs, gate = inputs.to(device), gate.to(device)

                output = net(inputs, gate)

                if len(output_val) == 0:
                    output_val = output.cpu().detach().numpy()
                else:
                    output_val = np.concatenate((output_val, output.cpu().detach().numpy()), axis=0)

                # loss = criterion(output, targets)
                # val_loss += loss.item()
        # print(val_loss / len(test_loader.dataset))
        # return val_loss / len(test_loader.dataset), np.array(target_val), np.array(output_val)
        if (phase == "stance"):
            output_val = np.array(output_val)
            data = scaler_stance_post.inverse_transform(output_val.reshape(-1, output_val.shape[-1])).reshape(
                output_val.shape)
        if (phase == "swing"):
            output_val = np.array(output_val)
            data = scaler_swing_post.inverse_transform(output_val.reshape(-1, output_val.shape[-1])).reshape(
                output_val.shape)
        return np.array(data)

def merge_st_sw_phase(st,sw,st_prop,sw_prop):
  x = np.linspace(0, 100, 51)

  st_prop = round((st_prop/100)*51)
  sw_prop = 51 - st_prop

  z = np.linspace(0, 100, st_prop)

  int_fun = interp1d(x, st, 'cubic', axis=0)
  st_phase = int_fun(z)

  z = np.linspace(0, 100, sw_prop)

  int_fun = interp1d(x, sw, 'cubic', axis=0)
  sw_phase = int_fun(z)

  return np.append(st_phase,sw_phase)

selectPatientBtn = Button(top, text="Select Patient", command=show)
selectPatientBtn.place(x=50, y=30)

muscleOptions = ["Yes", "No"]

leftSideLbt = Label(top, text = "Left Side")
leftSideLbt.place(x=170, y=60)

rightSideLbt = Label(top, text = "Right Side")
rightSideLbt.place(x=300, y=60)

muscle0Lbt = Label(top, text = "Muscle 0 Injected")
muscle0Lbt.place(x=50, y=90)
muscle0Cmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle0Cmb.set("Select Option")
muscle0Cmb.place(x=150, y=90)

muscle0RCmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle0RCmb.set("Select Option")
muscle0RCmb.place(x=280, y=90)

muscle1Lbt = Label(top, text = "Muscle 1 Injected")
muscle1Lbt.place(x=50, y=120)
muscle1Cmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle1Cmb.set("Select Option")
muscle1Cmb.place(x=150, y=120)

muscle1RCmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle1RCmb.set("Select Option")
muscle1RCmb.place(x=280, y=120)

muscle2Lbt = Label(top, text = "Muscle 2 Injected")
muscle2Lbt.place(x=50, y=150)
muscle2Cmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle2Cmb.set("Select Option")
muscle2Cmb.place(x=150, y=150)

muscle2RCmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle2RCmb.set("Select Option")
muscle2RCmb.place(x=280, y=150)

muscle3Lbt = Label(top, text = "Muscle 3 Injected")
muscle3Lbt.place(x=50, y=180)
muscle3Cmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle3Cmb.set("Select Option")
muscle3Cmb.place(x=150, y=180)

muscle3RCmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle3RCmb.set("Select Option")
muscle3RCmb.place(x=280, y=180)

muscle4Lbt = Label(top, text = "Muscle 4 Injected")
muscle4Lbt.place(x=50, y=210)
muscle4Cmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle4Cmb.set("Select Option")
muscle4Cmb.place(x=150, y=210)

muscle4RCmb = ttk.Combobox(top, values = muscleOptions, state="readonly", width=12)
muscle4RCmb.set("Select Option")
muscle4RCmb.place(x=280, y=210)

modelOptions = ["5-BiLSTMs", "Gated 5-BiLSTMs", "5-BiLSTMs with att mechanism"]
modelLbt = Label(top, text = "Deep Learning Model")
modelLbt.place(x=50, y=240)
muscle4Cmb = ttk.Combobox(top, values = modelOptions)
muscle4Cmb.set("Select Model")
muscle4Cmb.place(x=170, y=240)

selectPatientBtn = Button(top, text="Submit", command=show1)
selectPatientBtn.place(x=50, y=280)

top.mainloop()