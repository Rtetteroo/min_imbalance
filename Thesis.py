# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:33:49 2023

@author: Rogier
"""

import control 
import numpy as np

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
print(eig_A_c)
#poles = [9e-01, 2.44901876e-01, -0.0001, 9.96838172e-01, 0.45750675e-01, -0.0001]  
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
T_m = 85
T_M = 90
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

w_min = pwlinear_out_HW1_y2w(T_m) # minimum w corresponding to T 85 C 
w_max = pwlinear_out_HW1_y2w(T_M) # maximum w corresponding to T 90 C 

x_ = np.linspace(3,10,100)
y = np.array([pwlinear_in_HW1_u2h(x) for x in x_])
x_ = x_.reshape((-1,1))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_,y)
b = model.intercept_
a = model.coef_

#%%
Q_imb = np.random.uniform(-1,1, 50) * 24
#%%
from pyomo.environ import *

time = 12
states = 6
farad = 96485
z = 2

uti = 0.8
stor_cap = 6

# Chlor-alkali plant
nr_elec_c = 24
nr_cells_c = 160
area_c = 27000
j_max = 0.6
j_nom_c = j_max * uti
P_nom_c = -0.801429 + 15.5039 * j_nom_c
n_nom_c = nr_elec_c * nr_cells_c * j_nom_c * area_c / (z * farad)
stor_max_c = n_nom_c * 3600 * stor_cap

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

# Variables chlor-alkali electrolyser
T_in, T_mod_c, T_plant_c, P_c, j_mod_c, j_plant_c = [],[],[],[],[],[]
stor_c_model, stor_c_plant = [],[]
obj, Q_min, Q_plus = [],[],[]
T_mod_op_c = []
 
for i, imb in enumerate(Q_imb): 
    if i == 20: 
        break
    
    m = ConcreteModel()
     
    # Variables
    m.time = Set(initialize=range(time))
    m.time_plus1 = Set(initialize=range(time+1))
    m.states = Set(initialize=range(states))
    m.x_c = Var(m.time_plus1, m.states, within=Reals)
    m.P_c = Var(m.time, within=NonNegativeReals)
    m.Q_dr_c = Var(m.time, within=Reals)
    m.P_h_c = Var(m.time, within=Reals)
    m.T_in =  Var(m.time, within=NonNegativeReals)
    m.stor_c = Var(m.time_plus1, within=NonNegativeReals)
    m.stor_norm_c = Var(m.time_plus1, within=NonNegativeReals)
    m.T_w_c = Var(m.time_plus1, within=Reals)
    m.j_c = Var(m.time, within=Reals)
    m.N_dot_c = Var(m.time, within=Reals)
    m.Q_min = Var(m.time, within=NegativeReals)
    m.Q_plus = Var(m.time, within=NonNegativeReals)
    m.Q_slack = Var(m.time, within=NonNegativeReals)
    
    # Objective function
    def obj_rule(m):
        return sum(m.Q_plus[t] - m.Q_min[t] + m.Q_slack[t] for t in m.time) \
    
    m.obj = Objective(rule=obj_rule)  
    
    # Constraints
    m.cons = ConstraintList()
    [m.cons.add(m.x_c[0,s] == x0_c[s]) for s in m.states]
    m.cons.add(m.stor_c[0] == stor_max_c*0.5)
# =============================================================================
#     for t in m.time:
#         for s in m.states:
#             m.cons.add(m.x_c[t+1,s] == 
#                        sum(A_c[s,c]*m.x_c[t,c] for c in m.states) \
#                        + B_t[s,0] * m.P_h_c[t] + B_t[s,1]* m.T_in[t] \
#                        )
# =============================================================================
        
    def x_temp_c(m, t, s):
        return m.x_c[t+1,s] == sum(A_c[s,c]*m.x_c[t,c] for c in m.states) \
                                + B_t[s,0] * m.P_h_c[t] + B_t[s,1]* m.T_in[t] 
    
    m.cons_temp = Constraint(m.time, m.states, rule=x_temp_c)
    
    #m.con_HW1 = Piecewise(m.time, m.P_h_c, m.P_c, pw_pts = x_pw_in_c, pw_constr_type = 'EQ', f_rule = y_pw_in_c, pw_repn="BIGM_BIN")
    for t in m.time:
        m.cons.add(expr=(70,m.T_in[t],80))
        m.cons.add(m.Q_plus[t] <= abs(Q_imb[i+t])+m.Q_slack[t])
        m.cons.add(-abs(Q_imb[i+t])-m.Q_slack[t] <= m.Q_min[t])
        m.cons.add(Q_imb[i+t] == m.Q_plus[t] + m.Q_min[t] + m.Q_dr_c[t])
        m.cons.add(m.Q_dr_c[t] == nr_elec_c * 900 / 3600 * (P_nom_c - m.P_c[t]))
        m.cons.add(m.P_h_c[t] == a*m.P_c[t] + b)
        m.cons.add(m.P_c[t] == 15.5039 * m.j_c[t] - 0.801429)
        m.cons.add(expr=(0.3, m.j_c[t], 0.6))
        m.cons.add(m.N_dot_c[t] == nr_elec_c*nr_cells_c*area_c*m.j_c[t]/(z*farad))
        m.cons.add(m.stor_c[t+1] == m.stor_c[t] + m.N_dot_c[t]*900 - n_nom_c*900)
        if t > 0:
            m.cons.add(expr=(-0.075, m.j_c[t-1] - m.j_c[t], 0.075))
    
    for t in m.time_plus1:
        m.cons.add(m.T_w_c[t] == sum(C_c[0,c]*m.x_c[t,c] for c in m.states))
        m.cons.add(m.stor_norm_c[t] == m.stor_c[t]/stor_max_c)
        if t > 0:
            m.cons.add(expr=(0.2, m.stor_norm_c[t], 0.8))
            m.cons.add(expr=(w_min, m.T_w_c[t], w_max)) 
    
    solver = SolverFactory('appsi_highs') 
    solver.solve(m) 
    
    # Get inputs P,T_in & simulate FMU
    u_P_c,u_Ph_c,u_T_in = m.P_c[0].value, m.P_h_c[0].value, m.T_in[0].value       
    
    input_c = np.array([(0.0, u_P_c, u_T_in, T0_c, NaCl,NaOH,H2O_ano,H2O_cat,j_nom_c)], dtype = dtype_c)
    results_c = simulate_fmu(unzipdir_c,model_description=mod_descr_c, start_time=0,output_interval=50, stop_time=900, step_size=1, apply_default_start_values= True, input = input_c, output=output_c) 

    # State estimation CA electrolysis
    x_c = np.matrix([m.x_c[0,s].value for s in m.states]).T
    u_input_c = np.matrix([u_Ph_c, u_T_in]).T
    w0_mod_c = m.T_w_c[0].value
    w0_plant_c = pwlinear_out_HW1_y2w(T0_c)
    error = w0_plant_c - w0_mod_c
    x_new_est = A_c*x_c + B_c*u_input_c +L_c*error
    x0_c = x_new_est.T.tolist()[0]
    
    # Results
    T0_c = results_c['T'][-1]
    NaCl = results_c['n_moles_ano_cl'][-1]
    NaOH = results_c['n_moles_cat_oh'][-1]
    H2O_ano = results_c['n_moles_ano_h2o'][-1]
    H2O_cat = results_c['n_moles_cat_h2o'][-1]
    #%%
plt.plot(T_mod_op_c)
plt.plot(T_plant_c)
#%%

    # Append data
    T_mod_c.append(pwlinear_out_HW1_w2y(m.T_w_c[0].value))
    T_plant_c.append(T0_c),
    T_in.append(m.T_in[0].value)
    P_c.append(m.P_c[0].value)
    Q_min.append(model.Q_min[0].value)
    P_m_out.append(model.P_m_out[0].value) 
    i_den_model.append(model.i_den[0].value), i_den_plant.append(results['i_den'][0])
    Cl2_stor.append(Cl2_stor_init/Cl2_stor_max)    
    
# =============================================================================
#     for v in m.component_objects(Var, active=True):
#         print("Variable",v)  
#         for index in v:
#             print ("   ",index, value(v[index]))
#             
# =============================================================================
    obj.append(m.obj.expr())
    #%%
print(obj)
#%%
    # Append data
    T_model.append(pwlinear_out_HW1_w2y(model.w_t[0].value)), T_plant.append(T_0)
    T_in.append(model.T_in[0].value), P.append(model.P[0].value), P_m_in.append(model.P_m_in[0].value), P_m_out.append(model.P_m_out[0].value) 
    i_den_model.append(model.i_den[0].value), i_den_plant.append(results['i_den'][0])
    Cl2_stor.append(Cl2_stor_init/Cl2_stor_max)
    
    # State estimation
    x = np.matrix([model.x1[0].value, model.x2[0].value, model.x3[0].value,model.x4[0].value,model.x5[0].value,model.x6[0].value]).T
    u_input = np.matrix([u_P_h, u_T_in]).T
    w0_model = model.w_t[0].value
    w0_plant = pwlinear_out_HW1_y2w(T_0)
    error = w0_plant - w0_model
   # x_new_est = A_t*x + B_t*u_input
    x_new_est = A_t*x + B_t*u_input +L_t*error
   
    # Update states
    [x1_0, x2_0, x3_0, x4_0, x5_0, x6_0] = [x_new_est[i].item() for i in range(len(x_new_est))] 
   # Cl2_stor_init = model.cl2tot[1].value
    Cl2_stor_init = Cl2_stor_init + 24*results['S'][-1]-Cl2_cons
    T_0 = results['T'][-1]
    NaCl = results['n_moles_ano_cl'][-1]
    NaOH = results['n_moles_cat_oh'][-1]
    H2O_ano = results['n_moles_ano_h2o'][-1]
    H2O_cat = results['n_moles_cat_h2o'][-1]

for v in m.component_objects(Var, active=True):
    print("Variable",v)  
    for index in v:
        print ("   ",index, value(v[index]))
        
print(m.obj.expr())
print(perf_counter()- t)
print([pwlinear_out_HW1_w2y(value(m.T_w_c[t])) for t in m.time])
print([value(m.j_c[t]) for t in m.time])
print([value(m.stor_norm_c[t]) for t in m.time])
u_P = [value(m.P_c[t]) for t in m.time]
u_T_in = [value(m.T_in[t]) for t in m.time]

T_0_c = 85
NaCl = 345 
NaOH = 1139
H2O_ano = 5000
H2O_cat = 5500
i_nom = 0.8*0.6
T_fmu = []
for i in range(len(u_P)):
    input = np.array([(0.0, u_P[i], u_T_in[i], T_0_c, NaCl,NaOH,H2O_ano,H2O_cat,i_nom)], dtype = dtype)
    results = simulate_fmu(unzipdir,model_description=model_description, start_time=0,output_interval=10, stop_time=900, step_size=1, apply_default_start_values= False, input = input, output=output,solver='CVode')
    T_fmu.append(results['T'][0])
    T_0_c = results['T'][-1]
    NaCl = results['n_moles_ano_cl'][-1]
    NaOH = results['n_moles_cat_oh'][-1]
    H2O_ano = results['n_moles_ano_h2o'][-1]
    H2O_ano_list.append( results['n_moles_ano_cl'][0])
    H2O_cat = results['n_moles_cat_h2o'][-1]
print(Q_imb)

#%%

    
    
    m.P_m_in= Var(time, within=NonNegativeReals)
    m.P_m_out = Var(time, within=NonNegativeReals)
