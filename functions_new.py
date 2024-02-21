import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

def get_angles(EA_Filter, Events):
    nframes = len(EA_Filter['LPelvis']['value'])
    Left_Side_Angless = np.empty((nframes, 3))
    Right_Side_Angless = np.empty((nframes, 3))
    joint_angles = ['Pelvis', 'Hip', 'Knee', 'Ankle', 'Foot']
    one_right_cycle = []
    one_left_cycle = []
    i = 0
    j = 0
    freq = EA_Filter['LKnee']['Rate']
    for joint in joint_angles:
        l_joint = EA_Filter['L' + joint]
        l_joint_val = l_joint['value']
        if i == 0:
            Left_Side_Angless = l_joint_val.copy()
            i = i + 1
        else:
            Left_Side_Angless = np.concatenate((Left_Side_Angless, l_joint_val), axis=1)

        r_joint = EA_Filter['R' + joint]
        r_joint_val = r_joint['value']
        if j == 0:
            Right_Side_Angless = r_joint_val.copy()
            j = j + 1
        else:
            Right_Side_Angless = np.concatenate((Right_Side_Angless, r_joint_val), axis=1)

    right_IC, right_TSE, left_IC, left_TSE = get_events(Events,rate=freq)

    print('Left Side Shape : ', Left_Side_Angless.shape)

    for i in range(len(left_IC)):
        one_left_cycle.append(Left_Side_Angless[left_IC[i]:left_TSE[i]][:, [0, 1, 2, 3, 4, 5, 6, 9, 14]])

    print('Right side Shape : ', Right_Side_Angless.shape)

    for i in range(len(right_IC)):
        one_right_cycle.append(Right_Side_Angless[right_IC[i]:right_TSE[i]][:, [0, 1, 2, 3, 4, 5, 6, 9, 14]])

    return one_right_cycle, one_left_cycle


def get_events(Events, rate=100):
    # rate could be different to 100 Hz, so it is bette to keep it as variable with a default value of 100
    print("rate = ",rate)
    right_events = Events['Right']
    left_events = Events['Left']

    right_IC = []
    right_TSE = []
    left_IC = []
    left_TSE = []

    if type(right_events['IC']) == float:
        r_e_IC = right_events['IC']
        r_e_TSE = right_events['TSE']
        right_IC.append(int(r_e_IC * rate))
        right_TSE.append(int(r_e_TSE * rate))
        l_e_IC = left_events['IC']
        l_e_TSE = left_events['TSE']
        left_IC.append(int(l_e_IC * rate))
        left_TSE.append(int(l_e_TSE * rate))
    else:
        len_right_events = len(right_events['IC'])
        for i in range(len_right_events):

            r_e_IC = right_events['IC'][i]
            r_e_TSE = right_events['TSE'][i]

            right_IC.append(int(r_e_IC * rate))
            right_TSE.append(int(r_e_TSE * rate))


            l_e_IC = left_events['IC'][i]
            l_e_TSE = left_events['TSE'][i]

            left_IC.append(int(l_e_IC * rate))
            left_TSE.append(int(l_e_TSE * rate))

    return right_IC, right_TSE, left_IC, left_TSE


def cycle_normalization(Events, LCycle, RCycle, npoint=51,rate=100):
    #freq = EA_Filter['LKnee']['Rate']

    RStart, REnd, LStart, LEnd = get_events(Events,rate=rate)
    grr = []
    gll = []

    for i in range(len(LCycle)):
        RStartt = RStart[i]
        REndd = REnd[i]
        LStartt = LStart[i]
        LEndd = LEnd[i]
        # Start Parameters

        nl = LEndd - LStartt
        LeftCycle = np.linspace(0, 100, nl)
        nr = REndd - RStartt
        RightCycle = np.linspace(0, 100, nr)
        CompleteCycle = np.linspace(0, 100, npoint)
        # End Parameters

        # Left "Side Normalization
        print("Left Cycle = ",len(LeftCycle))
        print("LCycle[i] = ", len(LCycle[i]))

        lpoly = interp1d(LeftCycle, LCycle[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)

        gll.append(gl)

        rpoly = interp1d(RightCycle, RCycle[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr.append(gr)
    return gll, grr

def cycle_normalization_single_joint(Events, LCycle, RCycle, npoint=51):
    RStart, REnd, LStart, LEnd = get_events(Events)
    grr = []
    gll = []

    for i in range(len(LCycle)):
        RStartt = RStart[i]
        REndd = REnd[i]
        LStartt = LStart[i]
        LEndd = LEnd[i]
        # Start Parameters

        nl = LEndd - LStartt
        LeftCycle = np.linspace(0, 100, nl)
        nr = REndd - RStartt
        RightCycle = np.linspace(0, 100, nr)
        CompleteCycle = np.linspace(0, 100, npoint)
        # End Parameters

        # Left "Side Normalization

        lpoly = interp1d(LeftCycle, LCycle[i])
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)

        gll.append(gl)

        rpoly = interp1d(RightCycle, RCycle[i])
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr.append(gr)
    return gll, grr

def merge_pre_post_treatment(pre_tre,tre,post_tre):
    pre_tre_df = pd.read_csv(pre_tre)
    post_tre_df = pd.read_csv(post_tre)
    tre_df = pd.read_csv(tre)

    pre_post_df = pd.merge(pd.merge(pre_tre_df, tre_df, on='Patient Name'), post_tre_df, on='Patient Name')
    return (pre_post_df)

def get_ankle_knee_angles(EA_Filter, Events,rate=100):

    nframes = len(EA_Filter['LPelvis']['value'])
    Left_Side_Angless = np.empty((nframes, 3))
    Right_Side_Angless = np.empty((nframes, 3))
    joint_angles = ['Knee', 'Ankle']
    one_right_cycle = []
    one_left_cycle = []
    i = 0
    j = 0
    for joint in joint_angles:
        l_joint = EA_Filter['L' + joint]
        l_joint_val = l_joint['value']
        if i == 0:
            Left_Side_Angless = l_joint_val.copy()
            i = i + 1
        else:
            Left_Side_Angless = np.concatenate((Left_Side_Angless, l_joint_val), axis=1)

        r_joint = EA_Filter['R' + joint]
        r_joint_val = r_joint['value']
        if j == 0:
            Right_Side_Angless = r_joint_val.copy()
            j = j + 1
        else:
            Right_Side_Angless = np.concatenate((Right_Side_Angless, r_joint_val), axis=1)

    right_IC, right_TSE, left_IC, left_TSE = get_events(Events,rate)
    print('Left Side Shape : ', Left_Side_Angless.shape)

    for i in range(len(left_IC)):
        one_left_cycle.append(Left_Side_Angless[left_IC[i]:left_TSE[i]][:, [1, 4]])

    print('Right side Shape : ', Right_Side_Angless.shape)

    for i in range(len(right_IC)):
        one_right_cycle.append(Right_Side_Angless[right_IC[i]:right_TSE[i]][:, [1, 4]])

    """plt.title("Patient i Knee")
    plt.plot((Left_Side_Angless[:, [1]]), color='green')
    plt.plot((Right_Side_Angless[:, [1]]), color='orange')
    plt.legend(["Left Side", "Right Side"], loc=0)
    plt.show()

    plt.title("Patient i Ankle")
    plt.plot((Left_Side_Angless[:, [4]]), color='green')
    plt.plot((Right_Side_Angless[:, [4]]), color='orange')
    plt.legend(["Left Side", "Right Side"], loc=0)
    plt.show()"""

    return one_right_cycle, one_left_cycle

def get_knee_angles(EA_Filter, Events,rate=100):
    nframes = len(EA_Filter['LPelvis']['value'])
    Left_Side_Angless = np.empty((nframes, 3))
    Right_Side_Angless = np.empty((nframes, 3))
    joint_angles = ['Knee']
    one_right_cycle = []
    one_left_cycle = []
    i = 0
    j = 0
    for joint in joint_angles:
        l_joint = EA_Filter['L' + joint]
        l_joint_val = l_joint['value']
        if i == 0:
            Left_Side_Angless = l_joint_val.copy()
            i = i + 1
        else:
            Left_Side_Angless = np.concatenate((Left_Side_Angless, l_joint_val), axis=1)

        r_joint = EA_Filter['R' + joint]
        r_joint_val = r_joint['value']
        if j == 0:
            Right_Side_Angless = r_joint_val.copy()
            j = j + 1
        else:
            Right_Side_Angless = np.concatenate((Right_Side_Angless, r_joint_val), axis=1)

    right_IC, right_TSE, left_IC, left_TSE = get_events(Events,rate)

    print('Left Side Shape : ', len(Left_Side_Angless))
    Left_Side_Angless = np.transpose(Left_Side_Angless)
    for i in range(len(left_IC)):
        one_left_cycle.append(Left_Side_Angless[1][left_IC[i]:left_TSE[i]])

    print('Right side Shape : ', len(Right_Side_Angless))
    Right_Side_Angless = np.transpose(Right_Side_Angless)
    for i in range(len(right_IC)):
        one_right_cycle.append(Right_Side_Angless[1][right_IC[i]:right_TSE[i]])

    return one_right_cycle, one_left_cycle

def get_full_ankle_knee_angles(EA_Filter, Events):
    nframes = len(EA_Filter['LPelvis']['value'])
    Left_Side_Angless = np.empty((nframes, 3))
    Right_Side_Angless = np.empty((nframes, 3))
    joint_angles = ['Knee', 'Ankle']
    i = 0
    j = 0
    for joint in joint_angles:
        l_joint = EA_Filter['L' + joint]
        l_joint_val = l_joint['value']
        if i == 0:
            Left_Side_Angless = l_joint_val.copy()
            i = i + 1
        else:
            Left_Side_Angless = np.concatenate((Left_Side_Angless, l_joint_val), axis=1)

        r_joint = EA_Filter['R' + joint]
        r_joint_val = r_joint['value']
        if j == 0:
            Right_Side_Angless = r_joint_val.copy()
            j = j + 1
        else:
            Right_Side_Angless = np.concatenate((Right_Side_Angless, r_joint_val), axis=1)

    return Right_Side_Angless[:,[1,4]], Left_Side_Angless[:,[1,4]]