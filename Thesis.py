# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:33:49 2023

@author: Rogier
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import control
import time
from fmpy import *
from pyomo.environ import * 
from datetime import datetime
import pybind11
from highspy import *
import time

#%%

size = 300

file = r"C:\Users\Rogier\OneDrive\Documenten\Thesis\Python\Wind data.csv"
wind_data = pd.read_csv(file)
wind_DA = np.array(wind_data['DA wind']) * size
wind_DA_avg = np.array([sum(wind_DA[i*4:i*4+4])/4 for i in range(int(len(wind_data)/4)) for j in range(4)])
wind_act = np.array(wind_data['Actual wind'])*size

file = r"C:\Users\Rogier\OneDrive\Documenten\Thesis\Python\PV data.csv"
pv_data = pd.read_csv(file)
pv_DA = np.array(pv_data['DA PV']) * size
pv_DA_avg = np.array([sum(pv_DA[i*4:i*4+4])/4 for i in range(int(len(pv_data)/4)) for j in range(4)])
pv_act = np.array(pv_data['Actual PV'])*size
RE_act = pv_act+wind_act
RE_DA = pv_DA_avg + wind_DA_avg
tot_error = RE_act - RE_DA

RE_data = pd.DataFrame()
RE_data['datetime']  = pd.to_datetime(wind_data['DateTime'], format="%d/%m/%Y %H:%M")
RE_data = RE_data.set_index('datetime').sort_index()
RE_data['wind act'] = wind_act
RE_data['wind DA'] = wind_DA_avg
RE_data['pv act'] = pv_act
RE_data['pv DA'] = pv_DA_avg
RE_data['error'] = tot_error

fig,ax = plt.subplots()
x = np.linspace(1,8, len(RE_data['error'].loc['2020-10-01':'2020-10-07']))

plt.plot(x, np.array(RE_data['error'].loc['2020-10-01':'2020-10-07'])/4)
plt.show()

error_re = RE_data['error'].loc['2020-10-01':'2020-10-30'] / 4
Q_imb = error_re

print(sum(error_re[error_re>0]) - sum(error_re[error_re<0]))


#%%
A_c = np.matrix([[0.971440268930831, -0.475682434470563,0.074947008582989, 0, 0, 0],
                 [0.5,0 ,0, 0, 0,0],
                 [0, 0.0625, 0,0,0,0],
                 [0,0,0,0.778753670750431,-0.302075112310769, 0.066766349051151],
                 [0,0,0, 0.5, 0,0],
                 [0,0,0, 0,0.0625,0.004250766205142]])
B_c = np.matrix([[2, 0],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 0],
               [0, 0]])
C_c = np.matrix([0.500000000000000,  -0.517377124057972, 0, 1, -0.654714710207144, 0])
D_c = np.matrix([0,0])

eig_A_c = np.linalg.eigvals(A_c)

L_c = np.matrix(control.acker(np.transpose(A_c),np.transpose(C_c),eig_A_c)).transpose()
poles = [eig_A_c/10 for i in range(len(eig_A_c))]
poles = [0.30, 0.30, -0.01, 0.3, -0.02, 0.02]
#eig_A_t_1 = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
L_c = np.matrix(control.acker(np.transpose(A_c),np.transpose(C_c),poles)).transpose()

# Nonlinear Input block

x_pw_in_c = [1,5.702279759166746,9]
y_pw_in_c = [-85.509251359726079,-73.041698274944480,-62.23989946626932]
a_in_c = [0,0]
a_in_c[0] = (y_pw_in_c[1]-y_pw_in_c[0])/(x_pw_in_c[1]-x_pw_in_c[0])
a_in_c[1] = (y_pw_in_c[2]-y_pw_in_c[1])/(x_pw_in_c[2]-x_pw_in_c[1])
b_in_c = [y_pw_in_c[0], y_pw_in_c[1]]

def pwlinear_in_HW1_u2h(u):
    if u < x_pw_in_c[1]:
        h = a_in_c[0]*(u-x_pw_in_c[0]) + b_in_c[0]
    elif u >= x_pw_in_c[1]:
        h = a_in_c[1]*(u-x_pw_in_c[1]) + b_in_c[1]
    return h

def pwlinear_in_HW1_h2u(h):
    if h < y_pw_in_c[1]:
        u = (h-b_in_c[0])/a_in_c[0] + x_pw_in_c[0]
    elif y_pw_in_c[1] <= h <= y_pw_in_c[2]:
        u = (h-b_in_c[1])/a_in_c[1] + x_pw_in_c[1] 
    return u

# Nonlinear Output block
x_pw_out_c = [0,2.2768742466418,51.3745497108818]
y_pw_out_c = [84.462779691295196,85.7015496754111,111.4552907883152]
a_out_c = [0,0]
a_out_c[0] = (y_pw_out_c[1]-y_pw_out_c[0])/(x_pw_out_c[1]-x_pw_out_c[0])
a_out_c[1] = (y_pw_out_c[2]-y_pw_out_c[1])/(x_pw_out_c[2]-x_pw_out_c[1])
b_out_c = [y_pw_out_c[0], y_pw_out_c[1]]

def pwlinear_out_HW1_w2y(w):
    if w < x_pw_out_c[1]:
        y = a_out_c[0]*(w-x_pw_out_c[0]) + b_out_c[0]
    elif w >= x_pw_out_c[1]:
        y = a_out_c[1]*(w-x_pw_out_c[1]) + b_out_c[1]
    return y

def pwlinear_out_HW1_y2w(y): 
    if y < y_pw_out_c[1]:
        w = (y-b_out_c[0])/a_out_c[0] + x_pw_out_c[0]
    elif y >= y_pw_out_c[1]: 
        w = (y-b_out_c[1])/a_out_c[1] + x_pw_out_c[1]
    return w

x_ = np.linspace(3,10,100)
y = np.array([pwlinear_in_HW1_u2h(x) for x in x_])
x_ = x_.reshape((-1,1))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_,y)
b_c = model.intercept_
a_c = model.coef_

#%%

A_h = np.matrix([[1.400026330883430,  -0.465641052642110,  -0.121330119372427 , 0, 0, 0],
                 [1,0 ,0, 0, 0,0],
                 [0, 0.125, 0,0,0,0],
                 [0,0,0,1.519082149921198,-0.541285527501581,-0.137286030756005],
                 [0,0,0, 1.0, 0,0],
                 [0,0,0, 0,0.125,0]])
B_h = np.matrix([[2, 0],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 0],
               [0, 0]])
C_h = np.matrix([-0.650495698112021,  0.5, 0, 1, -0.887132010639724,0])
D_h = np.matrix([0,0])

eig_A_h = np.linalg.eigvals(A_h)
print(eig_A_h)
eig_A_h_1 =  [ 0.55, 0.55,  -0.0029837,    0.7,  0.6, -0.03]

L_h = np.matrix(control.acker(np.transpose(A_h),np.transpose(C_h),eig_A_h_1)).transpose()
#poles = [9e-01, 2.44901876e-01, -0.0001, 9.96838172e-01, 0.45750675e-01, -0.0001]  
#L_t = np.matrix(control.acker(np.transpose(A_t),np.transpose(C_t),eig_A_t)).transpose()

# Nonlinear Input block

x_pw_in_h = [0,6.275700648444316,9]
y_pw_in_h = [-10.395619175518608,-48.166754582244693,-62.539036863348535]
a_in_h = [0,0]
a_in_h[0] = (y_pw_in_h[1]-y_pw_in_h[0])/(x_pw_in_h[1]-x_pw_in_h[0])
a_in_h[1] = (y_pw_in_h[2]-y_pw_in_h[1])/(x_pw_in_h[2]-x_pw_in_h[1])
b_in_h = [y_pw_in_h[0], y_pw_in_h[1]]

def pwlinear_in_HW1_H2_u2h(u):
    if u < x_pw_in_h[1]:
        h = a_in_h[0]*(u-x_pw_in_h[0]) + b_in_h[0]
    elif u >= x_pw_in_h[1]:
        h = a_in_h[1]*(u-x_pw_in_h[1]) + b_in_h[1]
    return h

def pwlinear_in_HW1_H2_hu(h):
    if h > y_pw_in_h[1]:
        u = (h-b_in_h[0])/a_in_h[0] + x_pw_in_h[0]
    elif y_pw_in_h[1] >= h >= y_pw_in_h[2]:
        u = (h-b_in_h[1])/a_in_h[1] + x_pw_in_h[1] 
    return u

# Nonlinear Input block2

x_pw_in_Q_h = [0,0.752945419898490, 0.901772364769605,  1.032503045220910,  1.159489077681015,1.279250234776616, 1.421755053954330,1.5]
y_pw_in_Q_h = [-21.725592474929730, -51.767624536025885, -57.285436026399324, -61.847652126636667, -65.574655425202593,-68.961711968407897, -72.867316712498749, -74.908090815563838]
a_in_Q_h = [0,0,0,0,0,0,0]
a_in_Q_h[0] = (y_pw_in_Q_h[1]-y_pw_in_Q_h[0])/(x_pw_in_Q_h[1]-x_pw_in_Q_h[0])
a_in_Q_h[1] = (y_pw_in_Q_h[2]-y_pw_in_Q_h[1])/(x_pw_in_Q_h[2]-x_pw_in_Q_h[1])
a_in_Q_h[2] = (y_pw_in_Q_h[3]-y_pw_in_Q_h[2])/(x_pw_in_Q_h[3]-x_pw_in_Q_h[2])
a_in_Q_h[3] = (y_pw_in_Q_h[4]-y_pw_in_Q_h[3])/(x_pw_in_Q_h[4]-x_pw_in_Q_h[3])
a_in_Q_h[4] = (y_pw_in_Q_h[5]-y_pw_in_Q_h[4])/(x_pw_in_Q_h[5]-x_pw_in_Q_h[4])
a_in_Q_h[5] = (y_pw_in_Q_h[6]-y_pw_in_Q_h[5])/(x_pw_in_Q_h[6]-x_pw_in_Q_h[5])
a_in_Q_h[6] = (y_pw_in_Q_h[7]-y_pw_in_Q_h[6])/(x_pw_in_Q_h[7]-x_pw_in_Q_h[6])

b_in_Q_h = y_pw_in_Q_h

def pwlinear_in_HW2_H2_u2h(u):
    if u < x_pw_in_Q_h[1]:
        h = a_in_Q_h[0]*(u-x_pw_in_Q_h[0]) + b_in_Q_h[0]
    elif x_pw_in_Q_h[1]<= u <= x_pw_in_Q_h[2]:
        h = a_in_Q_h[1]*(u-x_pw_in_Q_h[1]) + b_in_Q_h[1]
    elif x_pw_in_Q_h[2]<= u <= x_pw_in_Q_h[3]:
        h = a_in_Q_h[2]*(u-x_pw_in_Q_h[2]) + b_in_Q_h[2]
    elif x_pw_in_Q_h[3]<= u <= x_pw_in_Q_h[4]:
        h = a_in_Q_h[3]*(u-x_pw_in_Q_h[3]) + b_in_Q_h[3]
    elif x_pw_in_Q_h[4]<= u <= x_pw_in_Q_h[5]:
        h = a_in_Q_h[4]*(u-x_pw_in_Q_h[4]) + b_in_Q_h[4]
    elif x_pw_in_Q_h[5]<= u <= x_pw_in_Q_h[6]:
        h = a_in_Q_h[5]*(u-x_pw_in_Q_h[5]) + b_in_Q_h[5]
    elif u >= x_pw_in_Q_h[6]:
        h = a_in_Q_h[6]*(u-x_pw_in_Q_h[6]) + b_in_Q_h[6]
    return h

def pwlinear_in_HW2_H2_h2u(h):
    if h > y_pw_in_Q_h[1]:
        u = (h-b_in_Q_h[0])/a_in_Q_h[0] + x_pw_in_Q_h[0]
    elif y_pw_in_Q_h[1] >= h >= y_pw_in_Q_h[2]:
        u = (h-b_in_Q_h[1])/a_in_Q_h[1] + x_pw_in_Q_h[1] 
    elif y_pw_in_Q_h[2] >= h >= y_pw_in_Q_h[3]:
        u = (h-b_in_Q_h[2])/a_in_Q_h[2] + x_pw_in_Q_h[2] 
    elif y_pw_in_Q_h[3] >= h >= y_pw_in_Q_h[4]:
        u = (h-b_in_Q_h[3])/a_in_Q_h[3] + x_pw_in_Q_h[3] 
    elif y_pw_in_Q_h[4] >= h >= y_pw_in_Q_h[5]:
        u = (h-b_in_Q_h[4])/a_in_Q_h[4] + x_pw_in_Q_h[4] 
    elif y_pw_in_Q_h[5] >= h >= y_pw_in_Q_h[6]:
        u = (h-b_in_Q_h[5])/a_in_Q_h[5] + x_pw_in_Q_h[5] 
    elif y_pw_in_Q_h[6] >= h:
        u = (h-b_in_Q_h[6])/a_in_Q_h[6] + x_pw_in_Q_h[6] 
    return u

# Nonlinear Output block
x_pw_out_h = [-3.0267232456631,5.4780595979014,6]
y_pw_out_h = [325.5115110841160,333.1871635413831,333.5388787225403]
a_out_h = [0,0]
a_out_h[0] = (y_pw_out_h[1]-y_pw_out_h[0])/(x_pw_out_h[1]-x_pw_out_h[0])
a_out_h[1] = (y_pw_out_h[2]-y_pw_out_h[1])/(x_pw_out_h[2]-x_pw_out_h[1])
b_out_h = [y_pw_out_h[0], y_pw_out_h[1]]

def pwlinear_out_HW1_H2_w2y(w):
    if w < x_pw_out_h[1]:
        y = a_out_h[0]*(w-x_pw_out_h[0]) + b_out_h[0]
    elif w >= x_pw_out_h[1]:
        y = a_out_h[1]*(w-x_pw_out_h[1]) + b_out_h[1]
    return y

def pwlinear_out_HW1_H2_y2w(y): 
    if y < y_pw_out_h[1]:
        w = (y-b_out_h[0])/a_out_h[0] + x_pw_out_h[0]
    elif y >= y_pw_out_h[1]: 
        w = (y-b_out_h[1])/a_out_h[1] + x_pw_out_h[1]
    return w

x_ = np.linspace(3,10,100)
y = np.array([pwlinear_in_HW1_H2_u2h(x) for x in x_])
x_ = x_.reshape((-1,1))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_,y)
b_h = model.intercept_
a_h = model.coef_

#%%

Time = 12
states = 6
farad = 96485
z = 2
uti = 0.8
stor_cap = 1
groups = 1
Q_imb = error_re

# Chlor-alkali plant
pen = [13-i for i in range(Time)]

groups_c = groups
nr_elec_c = 24
nr_cells_c = 160
area_c = 27000
j_min_c, j_max_c = 0.3, 0.6
j_ch_c = 0.075
j_nom_c = j_max_c * uti
P_nom_c = -0.801429 + 15.5039 * j_nom_c
Ndot_nom_c = nr_elec_c * nr_cells_c * j_nom_c * area_c / (z * farad)
T_min_c, T_max_c = 87.5, 87.5
w_min_c, w_max_c = pwlinear_out_HW1_y2w(T_min_c), pwlinear_out_HW1_y2w(T_max_c)
T_in_min, T_in_max = 70, 80
stor_max_c = Ndot_nom_c * 3600 * stor_cap
stor_init_c = 0.5 * stor_max_c
stor_init_plant_c = stor_init_c

# Hydrogen plant
groups_h = groups
nr_elec_h = 24
nr_cells_h = 60
nr_units_h = 160
area_h = 290
j_min_h, j_max_h = 1, 1.6
j_ch_h = 0.3
j_nom_h = j_max_h * uti
P_nom_h = -0.585551 + 5.40324 *j_nom_h
Ndot_nom_h = nr_elec_h * nr_cells_h * nr_units_h * j_nom_h * area_h / (z * farad)
Qhcool_min, Qhcool_max = pwlinear_in_HW2_H2_u2h(0),pwlinear_in_HW2_H2_u2h(2)
T_min_h, T_max_h = 57.5+273.15, 57.5+273.15
w_min_h, w_max_h = pwlinear_out_HW1_H2_y2w(T_min_h), pwlinear_out_HW1_H2_y2w(T_max_h)
stor_max_h = Ndot_nom_h * 3600 * stor_cap
stor_init_h = 0.5 * stor_max_h

fmu_c = r"C:\Users\Rogier\OneDrive\Documenten\Thesis\Python\CA_P_Tin_fmu.fmu"
unzipdir_c = extract(fmu_c)
mod_descr_c = read_model_description(unzipdir_c)
instantiate_fmu(unzipdir=unzipdir_c, model_description=mod_descr_c)

dtype_c =[('time', np.double), ('P_el_input', np.double), ('T_in',np.double), ('T_0', np.double), ('NaCl_0', np.double), ('NaOH_0', np.double), ('H2O_ano', np.double), ('H2O_cat', np.double), ('i_nom', np.double)]
output_c = ['T', 'n_moles_ano_cl', 'n_moles_cat_oh','n_moles_ano_h2o', 'n_moles_cat_h2o', 'i_den', 'BigNdot_cl2', 'S']

# Initial states
T0_c = 85
NaCl = 345 
NaOH = 1139
H2O_ano = 5000
H2O_cat = 5500

# Initial states HW model 1
x1_0,x2_0,x3_0,x4_0,x5_0,x6_0 = -527.69417107,-264.46795242,-16.54742593,190.57297152,95.56555782,6.00676246
x0_c = [-527.69417107,-264.46795242,-16.54742593,190.57297152,95.56555782,6.00676246]

# PWlinear INPUT breakpoints - P vs T
x_pw_in_c = [1,5.702279759166746,9]
y_pw_in_c = [-85.509251359726079,-73.041698274944480,-62.23989946626932]

fmu_h = r"C:\Users\Rogier\OneDrive\Documenten\Thesis\Python\PEM_PQ_FMU.fmu"
unzipdir_h = extract(fmu_h)
mod_descr_h = read_model_description(unzipdir_h)
instantiate_fmu(unzipdir=unzipdir_h, model_description=mod_descr_h)

dtype_h =[('time', np.double), ('P_el_input', np.double), ('Q_cool_input',np.double), ('T_0', np.double)]
output_h = ['T_op', 'S_H2','i_dens_a']

# Initial conditions FMU 

T0_h = 273.15+57.5

# Initial states HW model 1
x0_h = [-1314.502205480787, -1314.42682132694, -161.79160199074838, -1681.0979277342738, -1674.9481703798529, -207.0932507073316]

# PWlinear INPUT breakpoints - P vs T
x_pw_in_h = [0,6.275700648444316,9]
y_pw_in_h = [-10.395619175518608,-48.166754582244693,-62.539036863348535]

# Variables chlor-alkali electrolyser
T_in = []
T_mod_c = []
T_plant_c = []
P_c = []
P_agg_c = []
j_mod_c = []
j_plant_c = []
stor_model_c = []
stor_plant_c = []

T_mod_h = []
T_plant_h = []
P_h = []
P_agg_h = []
Q_cool = []
j_mod_h = []
j_plant_h = []
stor_model_h = []
stor_plant_h = []

obj = []
Q_res_imb = []
Q_min = []
Q_plus = []
Q_slack = []
t_sim = []

for i, imb in enumerate(Q_imb): 
    if i == 100: 
        break
    
    m = ConcreteModel()
     
    # Variables
    m.time = Set(initialize=range(Time))
    m.time_plus1 = Set(initialize=range(Time+1))
    m.states = Set(initialize=range(states))
    
    m.gr_c = Set(initialize=range(groups_c))
    m.x_c = Var(m.time_plus1, m.states, within=Reals)
    m.P_c = Var(m.time, within=NonNegativeReals, bounds=(3.5,8.5))
    m.Q_dr_c = Var(m.time, within=Reals)
    m.P_h_c = Var(m.time, within=Reals)
    m.T_in =  Var(m.time, within=NonNegativeReals)
    m.stor_c = Var(m.time_plus1, within=NonNegativeReals)
    m.stor_norm_c = Var(m.time_plus1, within=NonNegativeReals)
    m.T_w_c = Var(m.time_plus1, within=Reals)
    m.j_c = Var(m.time, within=Reals)
    m.N_dot_c = Var(m.time, within=Reals)
    
    m.gr_h = Set(initialize=range(groups_h))
    m.x_h = Var(m.time_plus1, m.states, within=Reals)
    m.P_h = Var(m.time, within=NonNegativeReals, bounds=(3.5,8.5))
    m.Q_dr_h = Var(m.time, within=Reals)
    m.P_h_h = Var(m.time, within=Reals)
    m.Qcool_h = Var(m.time, within=Reals)
    m.stor_h = Var(m.time_plus1, within=NonNegativeReals)
    m.stor_norm_h = Var(m.time_plus1, within=NonNegativeReals)
    m.T_w_h = Var(m.time_plus1, within=Reals)
    m.j_h = Var(m.time, within=Reals)
    m.N_dot_h = Var(m.time, within=Reals)    
    
    m.Q_min = Var(m.time, within=NegativeReals)
    m.Q_plus = Var(m.time, within=NonNegativeReals)
    m.Q_res_imb = Var(m.time, within=Reals)
    m.Q_slack = Var(m.time, within=NonNegativeReals)
    m.stor_delta = Var(m.time_plus1, within=Reals)
    
    # Objective function
    def obj_rule(m):
        return sum(m.Q_res_imb[t]**2 + m.stor_delta[t]**2 + m.Q_slack[t] for t in m.time) \
 #       return sum(10*(m.Q_plus[t]-m.Q_min[t])+ m.stor_delta[t]**2 + m.Q_slack[t] for t in m.time) \

    m.obj = Objective(rule=obj_rule)  
    
    # Constraints
    m.cons = ConstraintList()
    [m.cons.add(m.x_c[0,s] == x0_c[s]) for s in m.states]
    m.cons.add(m.stor_c[0] == stor_init_c)
    [m.cons.add(m.x_h[0,s] == x0_h[s]) for s in m.states]
    m.cons.add(m.stor_h[0] == stor_init_h)
    if i > 0:
        m.cons.add(expr=(-j_ch_c, j_plant_c[-1] - m.j_c[0], j_ch_c))
        m.cons.add(expr=(-j_ch_h, j_plant_h[-1] - m.j_h[0], j_ch_h))

    def x_temp_c(m, t, s):
        return m.x_c[t+1,s] == sum(A_c[s,c]*m.x_c[t,c] for c in m.states) \
                                + B_c[s,0] * m.P_h_c[t] + B_c[s,1]* m.T_in[t] 
    
    m.cons_x_temp_c = Constraint(m.time, m.states, rule=x_temp_c)
    
    def x_temp_h(m, t, s):
        return m.x_h[t+1,s] == sum(A_h[s,c]*m.x_h[t,c] for c in m.states) \
                                + B_h[s,0] * m.P_h_h[t] + B_h[s,1]*m.Qcool_h[t] 
    
    m.cons_x_temp_h = Constraint(m.time, m.states, rule=x_temp_h)
    
    m.con_HW1 = Piecewise(m.time, m.P_h_c, m.P_c, pw_pts = x_pw_in_c, pw_constr_type = 'EQ', f_rule = y_pw_in_c, pw_repn="SOS2")
    m.con_HW2 = Piecewise(m.time, m.P_h_h, m.P_h, pw_pts = x_pw_in_h, pw_constr_type = 'EQ', f_rule = y_pw_in_h, pw_repn="SOS2")
    for t in m.time:
        m.cons.add(expr=(T_in_min,m.T_in[t],T_in_max))
        m.cons.add(expr=(Qhcool_max,m.Qcool_h[t],Qhcool_min))
        m.cons.add(m.Q_res_imb[t] <= abs(Q_imb[i+t])+m.Q_slack[t])
        m.cons.add(-abs(Q_imb[i+t])-m.Q_slack[t] <= m.Q_res_imb[t])
        m.cons.add(Q_imb[i+t] == m.Q_res_imb[t] + m.Q_dr_c[t] + m.Q_dr_h[t])
#        m.cons.add(m.Q_plus[t] <= abs(Q_imb[i+t])+m.Q_slack[t])
#        m.cons.add(-abs(Q_imb[i+t])-m.Q_slack[t] <= m.Q_min[t])
#        m.cons.add(Q_imb[i+t] == m.Q_plus[t] - m.Q_min[t] + m.Q_dr_c[t] + m.Q_dr_h[t])
        m.cons.add(m.Q_dr_c[t] == nr_elec_c * (P_nom_c - m.P_c[t]) * 900 / 3600 )
        m.cons.add(m.Q_dr_h[t] == nr_elec_h * (P_nom_h - m.P_h[t]) * 900 / 3600 )
   #     m.cons.add(m.P_h_c[t] == a_c*m.P_c[t] + b_c)
   #     m.cons.add(m.P_h_h[t] == a_h*m.P_h[t] + b_h)
        m.cons.add(m.P_c[t] == 15.5039 * m.j_c[t] - 0.801429)
        m.cons.add(m.P_h[t] == 5.40324  * m.j_h[t] - 0.585551)  
        m.cons.add(expr=(j_min_c, m.j_c[t], j_max_c))
        m.cons.add(expr=(j_min_h, m.j_h[t], j_max_h))
        m.cons.add(m.N_dot_c[t] == nr_elec_c*nr_cells_c*area_c*m.j_c[t]/(z*farad))
        m.cons.add(m.N_dot_h[t] == nr_units_h*nr_elec_h*nr_cells_h*area_h*m.j_h[t]/(z*farad))
        m.cons.add(m.stor_c[t+1] == m.stor_c[t] + m.N_dot_c[t]*900 - Ndot_nom_c*900)
        m.cons.add(m.stor_h[t+1] == m.stor_h[t] + m.N_dot_h[t]*900 - Ndot_nom_h*900)
        if t > 0:
            m.cons.add(expr=(-j_ch_c, m.j_c[t-1] - m.j_c[t], j_ch_c))
            m.cons.add(expr=(-j_ch_h, m.j_h[t-1] - m.j_h[t], j_ch_h))
    
    for t in m.time_plus1:
        m.cons.add(m.T_w_c[t] == sum(C_c[0,c]*m.x_c[t,c] for c in m.states))
        m.cons.add(m.T_w_h[t] == sum(C_h[0,c]*m.x_h[t,c] for c in m.states))
        m.cons.add(m.stor_norm_c[t] == m.stor_c[t]/stor_max_c)
        m.cons.add(m.stor_norm_h[t] == m.stor_h[t]/stor_max_h)
        m.cons.add(m.stor_delta[t] == m.stor_norm_c[t] - m.stor_norm_h[t])
        if t > 0:
            m.cons.add(expr=(0.2, m.stor_norm_c[t], 0.8))
            m.cons.add(expr=(0.2, m.stor_norm_h[t], 0.8))
            m.cons.add(expr=(w_min_c, m.T_w_c[t], w_max_c)) 
            m.cons.add(expr=(w_min_h, m.T_w_h[t], w_max_h)) 

    solver = SolverFactory('gurobi', solver_io='python')
    
    t1 = time.perf_counter()
    solver.solve(m) 
    print(time.perf_counter()-t1)

    # Get inputs P,T_in & simulate FMU
    u_P_c,u_Ph_c,u_T_in = m.P_c[0].value, m.P_h_c[0].value, m.T_in[0].value       
    t1 = time.perf_counter()
    input_c = np.array([(0.0, u_P_c, u_T_in, T0_c, NaCl,NaOH,H2O_ano,H2O_cat,j_nom_c)], dtype = dtype_c)
    results_c = simulate_fmu(unzipdir_c,model_description=mod_descr_c, start_time=0,output_interval=50, stop_time=900, step_size=1, apply_default_start_values= True, input = input_c, output=output_c) 
    print(time.perf_counter()-t1)

    # State estimation CA electrolysis
    x_c = np.matrix([m.x_c[0,s].value for s in m.states]).T
    u_input_c = np.matrix([u_Ph_c, u_T_in]).T
    w0_mod_c = m.T_w_c[0].value
    w0_plant_c = pwlinear_out_HW1_y2w(T0_c)
    error = w0_plant_c - w0_mod_c
    x_new_c = A_c*x_c + B_c*u_input_c +L_c*error
    x0_c = x_new_c.T.tolist()[0]
    #x0_c = [m.x_c[1,s].value for s in m.states]
    T0_mod_c = pwlinear_out_HW1_w2y(w0_mod_c)
    
    # Get inputs P,T_in & simulate FMU
    u_P_h,u_Ph_h,u_Q_cool = m.P_h[0].value, m.P_h_h[0].value, pwlinear_in_HW2_H2_h2u(m.Qcool_h[0].value)       
    
    input_h = np.array([(0.0, u_P_h, u_Q_cool, T0_h)], dtype = dtype_h)
    results_h = simulate_fmu(unzipdir_h,model_description=mod_descr_h, start_time=0,output_interval=50, stop_time=900, step_size=1, apply_default_start_values= True, input = input_h, output=output_h)
    
    # State estimation CA electrolysis
    x_h = np.matrix([m.x_h[0,s].value for s in m.states]).T
    u_input_h = np.matrix([u_Ph_h, m.Qcool_h[0].value]).T
    w0_mod_h = m.T_w_h[0].value
    w0_plant_h = pwlinear_out_HW1_H2_y2w(T0_h)
    error = w0_plant_h - w0_mod_h
    x_new_h = A_h*x_h + B_h*u_input_h +L_h*error
    x0_h = x_new_h.T.tolist()[0]
   # x0_c = [m.x_c[1,s].value for s in m.states]
    T0_mod_h = pwlinear_out_HW1_H2_w2y(w0_mod_h)
    
    t_sim +=        [i]
    T_mod_c +=      [T0_mod_c]
    T_plant_c +=    [T0_c]
    T_in +=         [m.T_in[0].value]
    P_c +=          [m.P_c[0].value]
    P_agg_c +=      [m.P_c[0].value*nr_elec_c]
    j_mod_c +=      [m.j_c[0].value]
    j_plant_c +=    [results_c['i_den'][0]]
    stor_model_c += [m.stor_norm_c[0].value]
    stor_plant_c == [m.stor_norm_c[0].value]
    
    T_mod_h +=      [T0_mod_h] 
    T_plant_h +=    [T0_h]
    Q_cool +=       [u_Q_cool]
    P_h +=          [m.P_h[0].value]
    P_agg_h +=      [m.P_c[0].value*nr_elec_c]
    j_mod_h +=      [m.j_h[0].value]
    j_plant_h +=    [results_h['i_dens_a'][0]/10000]
    stor_model_h += [m.stor_norm_h[0].value]
    stor_plant_h += [m.stor_norm_h[0].value]
    
    obj +=          [m.obj.expr()]
    Q_res_imb +=    [m.Q_res_imb[0].value]    
    Q_min +=        [m.Q_min[0].value]  
    Q_plus +=       [m.Q_plus[0].value] 
    Q_slack +=      [m.Q_slack[0].value]

    T0_c = results_c['T'][-1]
    NaCl = results_c['n_moles_ano_cl'][-1]
    NaOH = results_c['n_moles_cat_oh'][-1]
    H2O_ano = results_c['n_moles_ano_h2o'][-1]
    H2O_cat = results_c['n_moles_cat_h2o'][-1]
    stor_init_c = stor_init_c + 24*results_c['S'][-1]-Ndot_nom_c*900

    T0_h = results_h['T_op'][-1]
    stor_init_h = stor_init_h + results_h['S_H2'][-1]*24 - Ndot_nom_h*900
    
#%%
plt.step(t_sim,P_h)
plt.step(t_sim,P_c)
plt.show()

plt.plot(Q_res_imb)
plt.plot(Q_slack)
plt.show()

plt.plot(stor_model_h)
plt.plot(stor_model_c)
plt.show()

plt.plot(T_mod_c)
plt.plot(T_plant_c)
plt.show()

plt.plot(T_mod_h)
plt.plot(T_plant_h)
plt.show()

plt.step(t_sim, Q_imb[:len(Q_res_imb)])
