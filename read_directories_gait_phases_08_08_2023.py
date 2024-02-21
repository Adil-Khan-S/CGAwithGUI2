import os

import functions_gait_phases_29_11_2022 as functions_new
import pandas as pd
from mat4py import loadmat

pre_treat_df = pd.DataFrame()
pre_treat_df_all_cycles = pd.DataFrame()
post_treat_df = pd.DataFrame()
post_treat_df_all_cycles = pd.DataFrame()
treatment_df = pd.DataFrame()
treatment_df_norm = pd.DataFrame()
treatment_df_all_cycles = pd.DataFrame()
treatment_df_norm_all_cycles = pd.DataFrame()
directory = "D:\PhD Material\Clinical Gait Analysis Data\MATLAB TOXINE"
df = pd.read_excel('D:\PhD Material\Clinical Gait Analysis Data\SUIVI TOXINE NEW.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'

folders = os.listdir(directory)
p_i = 1
for subdirectory in folders:
    tot_st_prop = 0
    tot_sw_prop = 0
    tot_sw_prop = 0
    print('Folder Name : ', subdirectory)
    pat_name = str(p_i)
    nom = subdirectory.split('_')[0]
    prenom = subdirectory.split('_')[1]
    row = df.loc[(df['Nom'].str.rstrip()+"".rstrip() == nom) & (df['Prenom'].str.rstrip()+"".rstrip() == prenom) ]
    c_n = row.columns.values.tolist()[11:30]
    c_n = [x.replace(' G', '') for x in c_n]

    l_t_d = row.iloc[:, df.columns.str.endswith(' G')]
    l_t_d.columns = c_n
    l_t_d.insert(0, "Patient Name", pat_name + " Left Side", True)
    treatment_df = treatment_df.append(l_t_d, ignore_index=True)

    r_t_d = row.iloc[:, df.columns.str.endswith(' D')]
    r_t_d.columns = c_n
    r_t_d.insert(0, "Patient Name", pat_name + " Right Side", True)
    treatment_df = treatment_df.append(r_t_d, ignore_index=True)

    subfolders = os.listdir(os.path.join(directory, subdirectory))
    pre_path = directory+"\\"+subdirectory+"\\PRE"
    files = os.listdir(pre_path)
    total_pre_cycles = 1
    for file in files:
        path = directory+"\\"+subdirectory+"\\PRE\\"+file
        print('File : ',path)
        mat = loadmat(path)
        EA_Filter = mat['EAFilter']
        Events = mat['Events']
        freq = EA_Filter['LKnee']['Rate']

        right_cycle, right_cycle_stance, right_cycle_swing, left_cycle, left_cycle_stance, left_cycle_swing = functions_new.get_ankle_knee_angles(EA_Filter, Events,rate=freq)

        left_interp_cycle, left_interp_stance_cycle, left_interp_swing_cycle, right_interp_cycle, right_interp_stance_cycle, right_interp_swing_cycle, st_prop, sw_prop = functions_new.cycle_normalization(Events, right_cycle, right_cycle_stance, right_cycle_swing, left_cycle, left_cycle_stance, left_cycle_swing, rate=freq)

        for i in range(len(left_interp_cycle)):
            #Left Side Stance Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Left"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Stance"

            for index, value in enumerate(left_interp_stance_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            l_t_d = row.iloc[:, df.columns.str.endswith(' G')]
            l_t_d.columns = c_n
            l_t_d.insert(0, "Patient_No", pat_name, True)
            l_t_d.insert(1, "Side", "Left", True)
            l_t_d.insert(2, "Cycle", total_pre_cycles, True)
            l_t_d.insert(3, "Phase", "Stance", True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(l_t_d, ignore_index=True)

            # Left Side Swing Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Left"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Swing"

            for index, value in enumerate(left_interp_swing_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            l_t_d = row.iloc[:, df.columns.str.endswith(' G')]
            l_t_d.columns = c_n
            l_t_d.insert(0, "Patient_No", pat_name, True)
            l_t_d.insert(1, "Side", "Left", True)
            l_t_d.insert(2, "Cycle", total_pre_cycles, True)
            l_t_d.insert(3, "Phase", "Swing", True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(l_t_d, ignore_index=True)

            #Right Side
            print('Right side shape after interpolation : ',right_interp_cycle[i].shape)
            # Right Side Stance Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Right"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Stance"

            for index, value in enumerate(right_interp_stance_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            r_t_d = row.iloc[:, df.columns.str.endswith(' D')]
            r_t_d.columns = c_n
            r_t_d.insert(0, "Patient_No", pat_name, True)
            r_t_d.insert(1, "Side", "Right", True)
            r_t_d.insert(2, "Cycle", total_pre_cycles, True)
            r_t_d.insert(3, "Phase", "Stance", True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(r_t_d, ignore_index=True)

            # Right Side Swing Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Right"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Swing"

            for index, value in enumerate(right_interp_swing_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            pre_treat_df_all_cycles = pre_treat_df_all_cycles.append(new_row, ignore_index=True)

            r_t_d = row.iloc[:, df.columns.str.endswith(' D')]
            r_t_d.columns = c_n
            r_t_d.insert(0, "Patient_No", pat_name, True)
            r_t_d.insert(1, "Side", "Right", True)
            r_t_d.insert(2, "Cycle", total_pre_cycles, True)
            r_t_d.insert(3, "Phase", "Swing", True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(r_t_d, ignore_index=True)

            total_pre_cycles = total_pre_cycles + 1
    #exit(0)
    pre_path = directory+"\\"+subdirectory+"\\POST"

    files = os.listdir(pre_path)
    total_pre_cycles = 1
    for file in files:
        path = directory+"\\"+subdirectory+"\\POST\\"+file
        print('File : ',path)
        mat = loadmat(path)
        EA_Filter = mat['EAFilter']
        Events = mat['Events']
        freq = EA_Filter['LKnee']['Rate']

        right_cycle, right_cycle_stance, right_cycle_swing, left_cycle, left_cycle_stance, left_cycle_swing = functions_new.get_ankle_knee_angles(
            EA_Filter, Events, rate=freq)

        left_interp_cycle, left_interp_stance_cycle, left_interp_swing_cycle, right_interp_cycle, right_interp_stance_cycle, right_interp_swing_cycle, st_prop, sw_prop = functions_new.cycle_normalization(
            Events, right_cycle, right_cycle_stance, right_cycle_swing, left_cycle, left_cycle_stance, left_cycle_swing,
            rate=freq)
        tot_st_prop += st_prop
        tot_sw_prop += sw_prop
        for i in range(len(left_interp_cycle)):
            # Left Side Stance Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Left"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Stance"

            for index, value in enumerate(left_interp_stance_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            post_treat_df_all_cycles = post_treat_df_all_cycles.append(new_row, ignore_index=True)

            """
            l_t_d = row.iloc[:, df.columns.str.endswith(' G')]
            l_t_d.columns = c_n
            l_t_d.insert(0, "Patient_No", pat_name, True)
            l_t_d.insert(1, "Side", "Left", True)
            l_t_d.insert(2, "Cycle", total_pre_cycles, True)

            treatment_df_all_cycles = treatment_df_all_cycles.append(l_t_d, ignore_index=True)
            """
            # Left Side Swing Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Left"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Swing"

            for index, value in enumerate(left_interp_swing_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            post_treat_df_all_cycles = post_treat_df_all_cycles.append(new_row, ignore_index=True)
            """
            l_t_d = row.iloc[:, df.columns.str.endswith(' G')]
            l_t_d.columns = c_n
            l_t_d.insert(0, "Patient_No", pat_name, True)
            l_t_d.insert(1, "Side", "Left", True)
            l_t_d.insert(2, "Cycle", total_pre_cycles, True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(l_t_d, ignore_index=True)
            """
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
            new_row = my_dict
            post_treat_df_all_cycles = post_treat_df_all_cycles.append(new_row, ignore_index=True)
            """
            r_t_d = row.iloc[:, df.columns.str.endswith(' D')]
            r_t_d.columns = c_n
            r_t_d.insert(0, "Patient_No", pat_name, True)
            r_t_d.insert(1, "Side", "Right", True)
            r_t_d.insert(2, "Cycle", total_pre_cycles, True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(r_t_d, ignore_index=True)
            """
            # Right Side Swing Phase
            my_dict = dict()
            my_dict["Patient_No"] = pat_name
            my_dict["Side"] = "Right"
            my_dict["Cycle"] = total_pre_cycles
            my_dict["Phase"] = "Swing"

            for index, value in enumerate(right_interp_swing_cycle[i].flatten(order='F')):
                my_dict[index + 1] = value
            new_row = my_dict
            post_treat_df_all_cycles = post_treat_df_all_cycles.append(new_row, ignore_index=True)
            """
            r_t_d = row.iloc[:, df.columns.str.endswith(' D')]
            r_t_d.columns = c_n
            r_t_d.insert(0, "Patient_No", pat_name, True)
            r_t_d.insert(1, "Side", "Right", True)
            r_t_d.insert(2, "Cycle", total_pre_cycles, True)
            treatment_df_all_cycles = treatment_df_all_cycles.append(r_t_d, ignore_index=True)
            """
            total_pre_cycles = total_pre_cycles + 1
    print("Tot Stance Prop of Patient ",tot_st_prop/len(files))
    print("Tot Swing Prop of Patient ", tot_sw_prop/len(files))

    p_i = p_i + 1
    #exit(0)
#These line is for saving Pre Treatment Data for all cycles
print('Pre Treatment datafreme of all cycles',pre_treat_df_all_cycles.iloc[0,1])
print('Shape pf Pre Treatment datafreme of all cycles',pre_treat_df_all_cycles.shape)
pre_treat_df_all_cycles.to_csv(r'D:\PhD Material\Clinical Gait Analysis Data\Generated Files Full Cycles With Phases AN 08_08_2023\pre_treat_df_all_cycles.csv', index=False)

#These line is for saving Post Treatment Data for all cycles
print('Post Treatment datafreme of all cycles ',post_treat_df_all_cycles)
print('Shape pf Post Treatment datafreme of all cycles',post_treat_df_all_cycles.shape)
post_treat_df_all_cycles.to_csv(r'D:\PhD Material\Clinical Gait Analysis Data\Generated Files Full Cycles With Phases AN 08_08_2023\post_treat_df_all_cycles.csv', index=False)

#These line is for saving Treatment Data Full Cycles without modification of columns into 0 and 1
treatment_df_all_cycles = treatment_df_all_cycles.fillna(0)
print(treatment_df_all_cycles)
print(treatment_df_all_cycles.shape)
treatment_df_all_cycles.to_csv(r'D:\PhD Material\Clinical Gait Analysis Data\Generated Files Full Cycles With Phases AN 08_08_2023\treatment_df_all_cycles.csv', index=False)

#These line is for saving Treatment Data Full Cycles after modification of columns into 0 and 1
treatment_df_norm_all_cycles = treatment_df_all_cycles
treatment_df_norm_col = treatment_df_norm_all_cycles.columns
treatment_df_norm_col = (pd.Series(treatment_df_norm_col))
treatment_df_norm_col = treatment_df_norm_col.drop([0,1,2,3])
for col in treatment_df_norm_col:
    treatment_df_norm_all_cycles.loc[treatment_df_norm_all_cycles[col] >= 1, col] = 1
treatment_df_norm_all_cycles.to_csv(r'D:\PhD Material\Clinical Gait Analysis Data\Generated Files Full Cycles With Phases AN 08_08_2023\treatment_df_norm_all_cycles.csv', index=False)