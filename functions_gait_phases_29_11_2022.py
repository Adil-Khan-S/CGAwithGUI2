import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def get_events(Events, rate=100):
    # rate could be different to 100 Hz, so it is bette to keep it as variable with a default value of 100
    #print("rate = ",rate)
    right_events = Events['Right']
    left_events = Events['Left']

    right_IC = []
    right_TO = []
    right_TSE = []
    left_IC = []
    left_TO = []
    left_TSE = []

    if type(right_events['IC']) == float:
        r_e_IC = right_events['IC']
        r_e_TO = right_events['TO']
        r_e_TSE = right_events['TSE']
        right_IC.append(int(r_e_IC * rate))
        right_TO.append(int(r_e_TO * rate))
        right_TSE.append(int(r_e_TSE * rate))
        l_e_IC = left_events['IC']
        l_e_TO = left_events['TO']
        l_e_TSE = left_events['TSE']
        left_IC.append(int(l_e_IC * rate))
        left_TO.append(int(l_e_TO * rate))
        left_TSE.append(int(l_e_TSE * rate))
    else:
        len_right_events = len(right_events['IC'])
        for i in range(len_right_events):

            r_e_IC = right_events['IC'][i]
            r_e_TO = right_events['TO'][i]
            r_e_TSE = right_events['TSE'][i]

            right_IC.append(int(r_e_IC * rate))
            right_TO.append(int(r_e_TO * rate))
            right_TSE.append(int(r_e_TSE * rate))


            l_e_IC = left_events['IC'][i]
            l_e_TO = left_events['TO'][i]
            l_e_TSE = left_events['TSE'][i]

            left_IC.append(int(l_e_IC * rate))
            left_TO.append(int(l_e_TO * rate))
            left_TSE.append(int(l_e_TSE * rate))
    return right_IC, right_TO, right_TSE, left_IC, left_TO, left_TSE


def cycle_normalization(Events, RCycle, RCycle_stance, RCycle_swing, LCycle, LCycle_stance, LCycle_swing, npoint=51,rate=100):
    RStart, RTO, REnd, LStart, LTO, LEnd = get_events(Events,rate=rate)
    #print(RStart)
    #print(RTO)
    #print(REnd)

    grr = []
    grr_stance = []
    grr_swing = []

    gll = []
    gll_stance = []
    gll_swing = []

    tot_st_prop_r = 0
    tot_st_prop_l = 0
    for i in range(len(LCycle)):
        RStartt = RStart[i]
        RTOO = RTO[i]
        REndd = REnd[i]
        LStartt = LStart[i]
        LTOO = LTO[i]
        LEndd = LEnd[i]

        tot = REndd - RStartt
        #print("Stance phase", (RTOO - RStartt) / tot * 100)
        #print("Swing phase", (REndd - RTOO) / tot * 100)
        st_prop_r = ((RTOO - RStartt) / tot * 100)
        tot_st_prop_r += st_prop_r
        st_prop_l = ((LTOO - LStartt) / tot * 100)
        tot_st_prop_l += st_prop_l

        # Start Parameters
        nl = LEndd - LStartt
        LeftCycle = np.linspace(0, 100, abs(nl))

        nl = LTOO - LStartt
        LeftCycle_stance = np.linspace(0, 100, abs(nl))

        nl = LEndd - LTOO
        LeftCycle_swing = np.linspace(0, 100, abs(nl))

        #print("REndd ", REndd)
        #print("RTOO ", RTOO)
        #print("RStartt ", RStartt)

        nr = REndd - RStartt
        RightCycle = np.linspace(0, 100, abs(nr))

        nr = RTOO - RStartt
        #print("RightCycle_stance ",nr)
        #print("RCycle_stance[i] ", len(RCycle_stance[i]))
        RightCycle_stance = np.linspace(0, 100, abs(nr))

        nr = REndd - RTOO
        RightCycle_swing = np.linspace(0, 100, abs(nr))

        CompleteCycle = np.linspace(0, 100, npoint)
        # End Parameters

        # Left Side Normalization

        lpoly = interp1d(LeftCycle, LCycle[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)
        gll.append(gl)

        lpoly = interp1d(LeftCycle_stance, LCycle_stance[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)
        gll_stance.append(gl)

        lpoly = interp1d(LeftCycle_swing, LCycle_swing[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)
        gll_swing.append(gl)

        rpoly = interp1d(RightCycle, RCycle[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr.append(gr)

        rpoly = interp1d(RightCycle_stance, RCycle_stance[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr_stance.append(gr)

        rpoly = interp1d(RightCycle_swing, RCycle_swing[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr_swing.append(gr)
    tot_st_prop_r = tot_st_prop_r/(len(RCycle))
    tot_st_prop_l = tot_st_prop_l/(len(LCycle))
    #print(tot_st_prop)
    #print(tot_sw_prop)
    return gll, gll_stance, gll_swing, grr, grr_stance, grr_swing, tot_st_prop_r, tot_st_prop_l


def cycle_normalization_phase_prop(Events, RCycle, RCycle_stance, RCycle_swing, LCycle, LCycle_stance, LCycle_swing, npoint=51,rate=100):
    RStart, RTO, REnd, LStart, LTO, LEnd = get_events(Events,rate=rate)
    #print(RStart)
    #print(RTO)
    #print(REnd)

    grr = []
    grr_stance = []
    grr_swing = []

    gll = []
    gll_stance = []
    gll_swing = []

    tot_st_prop_l = []
    tot_sw_prop_l = []
    tot_st_prop_r = []
    tot_sw_prop_r = []
    for i in range(len(LCycle)):
        RStartt = RStart[i]
        RTOO = RTO[i]
        REndd = REnd[i]
        LStartt = LStart[i]
        LTOO = LTO[i]
        LEndd = LEnd[i]

        tot = REndd - RStartt
        #print("Right Stance phase", (RTOO - RStartt) / tot * 100)
        #print("Right Swing phase", (REndd - RTOO) / tot * 100)
        st_prop = ((RTOO - RStartt) / tot * 100)
        tot_st_prop_r.append(st_prop)
        sw_prop = ((REndd - RTOO) / tot * 100)
        tot_sw_prop_r.append(sw_prop)

        #print("Left Stance phase", (LTOO - LStartt) / tot * 100)
        #print("Left Swing phase", (LEndd - LTOO) / tot * 100)
        st_prop = ((LTOO - LStartt) / tot * 100)
        tot_st_prop_l.append(st_prop)
        sw_prop = ((LEndd - LTOO) / tot * 100)
        tot_sw_prop_l.append(sw_prop)

        # Start Parameters
        nl = LEndd - LStartt
        LeftCycle = np.linspace(0, 100, abs(nl))

        nl = LTOO - LStartt
        LeftCycle_stance = np.linspace(0, 100, abs(nl))

        nl = LEndd - LTOO
        LeftCycle_swing = np.linspace(0, 100, abs(nl))

        #print("REndd ", REndd)
        #print("RTOO ", RTOO)
        #print("RStartt ", RStartt)

        nr = REndd - RStartt
        RightCycle = np.linspace(0, 100, abs(nr))

        nr = RTOO - RStartt
        #print("RightCycle_stance ",nr)
        #print("RCycle_stance[i] ", len(RCycle_stance[i]))
        RightCycle_stance = np.linspace(0, 100, abs(nr))

        nr = REndd - RTOO
        RightCycle_swing = np.linspace(0, 100, abs(nr))

        CompleteCycle = np.linspace(0, 100, npoint)
        # End Parameters

        # Left Side Normalization

        lpoly = interp1d(LeftCycle, LCycle[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)
        gll.append(gl)

        lpoly = interp1d(LeftCycle_stance, LCycle_stance[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)
        gll_stance.append(gl)

        lpoly = interp1d(LeftCycle_swing, LCycle_swing[i], 'cubic', axis=0)
        gl = np.transpose(lpoly(CompleteCycle))
        gl = np.transpose(gl)
        gll_swing.append(gl)

        rpoly = interp1d(RightCycle, RCycle[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr.append(gr)

        rpoly = interp1d(RightCycle_stance, RCycle_stance[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr_stance.append(gr)

        rpoly = interp1d(RightCycle_swing, RCycle_swing[i], 'cubic', axis=0)
        gr = np.transpose(rpoly(CompleteCycle))
        gr = np.transpose(gr)
        grr_swing.append(gr)
    #tot_st_prop = tot_st_prop/len(LCycle)
    #tot_sw_prop = tot_sw_prop/len(LCycle)
    #print(tot_st_prop_r)
    #print(tot_sw_prop_r)
    #print(tot_st_prop_l)
    #print(tot_sw_prop_l)

    return gll, grr, tot_st_prop_r, tot_sw_prop_r, tot_st_prop_l, tot_sw_prop_l

def get_ankle_knee_angles(EA_Filter, Events,rate=100):

    nframes = len(EA_Filter['LPelvis']['value'])
    Left_Side_Angless = np.empty((nframes, 3))

    Right_Side_Angless = np.empty((nframes, 3))

    joint_angles = ['Knee', 'Ankle']
    one_right_cycle = []
    one_right_cycle_stance = []
    one_right_cycle_swing = []

    one_left_cycle = []
    one_left_cycle_stance = []
    one_left_cycle_swing = []

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

    right_IC, right_TO, right_TSE, left_IC, left_TO, left_TSE = get_events(Events,rate)
    #print('Left Side Shape : ', Left_Side_Angless.shape)

    for i in range(len(left_IC)):
        one_left_cycle.append(Left_Side_Angless[left_IC[i]:left_TSE[i]][:, [1, 4]])
        one_left_cycle_stance.append(Left_Side_Angless[left_IC[i]:left_TO[i]][:, [1, 4]])
        one_left_cycle_swing.append(Left_Side_Angless[left_TO[i]:left_TSE[i]][:, [1, 4]])

    #print('Right side Shape : ', Right_Side_Angless.shape)

    for i in range(len(right_IC)):
        one_right_cycle.append(Right_Side_Angless[right_IC[i]:right_TSE[i]][:, [1, 4]])
        one_right_cycle_stance.append(Right_Side_Angless[right_IC[i]:right_TO[i]][:, [1, 4]])
        one_right_cycle_swing.append(Right_Side_Angless[right_TO[i]:right_TSE[i]][:, [1, 4]])

    return one_right_cycle, one_right_cycle_stance, one_right_cycle_swing, one_left_cycle, one_left_cycle_stance, one_left_cycle_swing