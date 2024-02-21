from tkinter.filedialog import *
from tkinter import *
from tkinter import ttk
import os
import pandas as pd
from mat4py import loadmat
import functions_gait_phases_29_11_2022 as functions_new

top = Tk()

top.geometry("500x350")
top.title("Post Gait CGA Simulator")
def show():
    global patFolder
    patFolder = askdirectory()

def show1():
    global treatment
    treatment = [0,0,0,0,0]
    if (muscle0Cmb.get() == "Yes"):
        treatment[0] = 1
    if (muscle1Cmb.get() == "Yes"):
        treatment[1] = 1
    if (muscle2Cmb.get() == "Yes"):
        treatment[2] = 1
    if (muscle3Cmb.get() == "Yes"):
        treatment[3] = 1
    if (muscle4Cmb.get() == "Yes"):
        treatment[4] = 1
    readMatFilesOfPatients()
def readMatFilesOfPatients():
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

            my_dict["t1"] = treatment[0]
            my_dict["t2"] = treatment[1]
            my_dict["t3"] = treatment[2]
            my_dict["t4"] = treatment[3]
            my_dict["t5"] = treatment[4]

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

            my_dict["t1"] = treatment[0]
            my_dict["t2"] = treatment[1]
            my_dict["t3"] = treatment[2]
            my_dict["t4"] = treatment[3]
            my_dict["t5"] = treatment[4]

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

            my_dict["t1"] = treatment[0]
            my_dict["t2"] = treatment[1]
            my_dict["t3"] = treatment[2]
            my_dict["t4"] = treatment[3]
            my_dict["t5"] = treatment[4]

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

            my_dict["t1"] = treatment[0]
            my_dict["t2"] = treatment[1]
            my_dict["t3"] = treatment[2]
            my_dict["t4"] = treatment[3]
            my_dict["t5"] = treatment[4]

            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            total_pre_cycles = total_pre_cycles + 1

        print(pre_treat_df_all_cycles)

        # saving the dataframe
        pre_treat_df_all_cycles.to_csv(patFolder+'\\patient1.csv')
selectPatientBtn = Button(top, text="Select Patient", command=show)
selectPatientBtn.place(x=50, y=50)

muscleOptions = ["Yes", "No"]

muscle0Lbt = Label(top, text = "Muscle 0 Injected")
muscle0Lbt.place(x=50, y=90)
muscle0Cmb = ttk.Combobox(top, values = muscleOptions)
muscle0Cmb.set("Select Option")
muscle0Cmb.place(x=150, y=90)

muscle1Lbt = Label(top, text = "Muscle 1 Injected")
muscle1Lbt.place(x=50, y=120)
muscle1Cmb = ttk.Combobox(top, values = muscleOptions)
muscle1Cmb.set("Select Option")
muscle1Cmb.place(x=150, y=120)

muscle2Lbt = Label(top, text = "Muscle 2 Injected")
muscle2Lbt.place(x=50, y=150)
muscle2Cmb = ttk.Combobox(top, values = muscleOptions)
muscle2Cmb.set("Select Option")
muscle2Cmb.place(x=150, y=150)

muscle3Lbt = Label(top, text = "Muscle 3 Injected")
muscle3Lbt.place(x=50, y=180)
muscle3Cmb = ttk.Combobox(top, values = muscleOptions)
muscle3Cmb.set("Select Option")
muscle3Cmb.place(x=150, y=180)

muscle4Lbt = Label(top, text = "Muscle 4 Injected")
muscle4Lbt.place(x=50, y=210)
muscle4Cmb = ttk.Combobox(top, values = muscleOptions)
muscle4Cmb.set("Select Option")
muscle4Cmb.place(x=150, y=210)

modelOptions = ["5-BiLSTMs", "Gated 5-BiLSTMs", "5-BiLSTMs with att mechanism"]
modelLbt = Label(top, text = "Deep Learning Model")
modelLbt.place(x=50, y=240)
muscle4Cmb = ttk.Combobox(top, values = modelOptions)
muscle4Cmb.set("Select Model")
muscle4Cmb.place(x=170, y=240)

selectPatientBtn = Button(top, text="Submit", command=show1)
selectPatientBtn.place(x=50, y=280)

top.mainloop()