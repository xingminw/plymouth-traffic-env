import numpy as np
# specify the traffic network in plymouth road
num_iterations = 1000
T = 100
N = 6 # number of intersetions
num_scenario = 10
# random_seed = [(seed+1)*10 for seed in range(10)]

C = [None] * N # cells in each intersection   all the cells
O = [None] * N # origins in each inter
D = [None] * N # destinations
BI = [None] * N # same as the ADMM paper boundary cells of inflow
BO = [None] * N
BI_IN = [None] * N
BO_IN = [None] * N
I1 = [None] * N #intersection cells
I2 = [None] * N
I3 = [None] * N
I4 = [None] * N
V = [None] * N #diverging cells  
M = [None] * N #merging cells
beta = [None] * N #ratio to turn left/right
proc = [None] * N #proceding cell relationships
pred = [None] * N #preceding cell relationships
Jam_N = [None] * N
Q = [None] * N
Demand = [None] * N


C[0] = list(range(80))
O[0] = [0, 68]
D[0] = [57, 67]
BI[0] = [29]
BO[0] = [28]
BI_IN[0] = [1]
BO_IN[0] = [1]
I1[0] = [10]
I2[0] = [11,46,47]
I3[0] = [78]
I4[0] = [79]
V[0] = [9, 45, 77]
M[0] = [12, 48, 58]
# beta[0] = np.array([[0.181, 0, 0.73], [0.819, 0.604, 0], [0, 0.396, 0.27]])
# beta[0] = np.array([[0.181, 0.604, 0.73], [0.819, 0.396, 0.27]])
beta[0] = [None]*num_scenario
for xi in range(num_scenario):
    beta[0][xi] = [None]*T
    for t in range(T):
        center_val_1 = [0.181, 0.604, 0.73]
        len_val = len(center_val_1)
        sample_lb_1 = [center_val_1[p]*0.7 for p in range(len_val)]
        sample_ub_1 = [(1-center_val_1[p])*0.3+center_val_1[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_1 = [max((random_x[p]-0.5)*2*(sample_lb_1[p]-center_val_1[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_1[p]-center_val_1[p]), 0) for p in range(len_val)]
        random_ratio_2 = [1-random_ratio_1[p] for p in range(len_val)]
        beta[0][xi][t] = np.array([random_ratio_1, random_ratio_2])
# for t in range(T):

proc[0] = {}
proc[0][10] = 58
proc[0][11] = 12
proc[0][46] = 48
proc[0][47] = 58
proc[0][78] = 12
proc[0][79] = 48
proc[0].update({c: c+1 for c in list(set(C[0])-set(I1[0])-set(I2[0])-set(I3[0])-set(I4[0]))})
pred[0] = {}
pred[0][10] = 9
pred[0][11] = 9
pred[0][46] = 45
pred[0][47] = 45
pred[0][78] = 77
pred[0][79] = 77
pred[0][12] = [11, 78]
pred[0][48] = [46, 79]
pred[0][58] = [10, 47]
pred[0].update({c: c-1 for c in list(set(C[0])-set(I1[0])-set(I2[0])-set(I3[0])-set(I4[0])-set(M[0]))})

Jam_N[0] = np.zeros(len(C[0]))
for c in O[0]+D[0]: 
    Jam_N[0][c] = 99999
for c in C[0][1:9] + C[0][12:45] + C[0][48:57]:
    Jam_N[0][c] = 8
for c in C[0][58:67] + C[0][69:77]:
    Jam_N[0][c] = 4

for c in I1[0] + I2[0] + I3[0] + I4[0]:
    Jam_N[0][c] = 4
Jam_N[0][11] = 8
Jam_N[0][46] = 8
Jam_N[0][9] = 12
Jam_N[0][45] = 12
Jam_N[0][77] = 8

Q[0] = Jam_N[0]/4
Q[0][0] = 2
Q[0][68] = 1
for c in D[0]:
    Q[0][c] = 99999
""" Demand[0] = np.zeros(len(O[0]))
Demand[0][0] = 857
Demand[0][1] = 200 """
Demand[0] = [None]*len(O[0])
Demand[0][0] = [None]*num_scenario
Demand[0][1] = [None]*num_scenario
center_mean_1 = 857/1800
center_mean_2 = 200/1800
mean_ub_1 = center_mean_1*1.3
mean_lb_1 = center_mean_1*0.7
mean_ub_2 = center_mean_2*1.3
mean_lb_2 = center_mean_2*0.7
for xi in range(num_scenario):
    # Demand[0][0][xi] = [None]*T
    # Demand[0][1][xi] = [None]*T
        # Demand[0][0][xi][t] = 857
        # Demand[0][1][xi][t] = 200
    mean_1 = np.random.random()*(mean_ub_1-mean_lb_1) + mean_lb_1
    mean_2 = np.random.random()*(mean_ub_2-mean_ub_2) + mean_lb_2
    Demand[0][0][xi] = np.random.poisson(mean_1, T)
    Demand[0][1][xi] = np.random.poisson(mean_2, T)
    

C[1] = list(range(112))
O[1] = [66, 89]
D[1] = [88, 111]
BI[1] = [0, 33]
BO[1] = [32, 65]
BI_IN[1] = [0, 2]
BO_IN[1] = [2, 0]
I1[1] = [16, 47]
I2[1] = [17,18,48,49]
I3[1] = [76, 99]
I4[1] = [77,78,100,101]
V[1] = [15,46,75,98]
M[1] = [19,50,79,102]
# beta[1] = np.array([[0.006,0.048,0.595,0.603],[0.954,0.928,0.055,0.111],[0.04,0.024,0.35,0.286]])
beta[1] = [None]*num_scenario
for xi in range(num_scenario):
    beta[1][xi] = [None]*T
    for t in range(T):
        center_val_1 = [0.006, 0.048, 0.595, 0.603] 
        len_val = len(center_val_1)
        sample_lb_1 = [center_val_1[p]*0.7 for p in range(len_val)]
        sample_ub_1 = [(1-center_val_1[p])*0.3+center_val_1[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_1 = [max((random_x[p]-0.5)*2*(sample_lb_1[p]-center_val_1[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_1[p]-center_val_1[p]), 0) for p in range(len_val)]

        center_val_2 = [0.954, 0.928, 0.055, 0.111]
        len_val = len(center_val_2)
        sample_lb_2 = [center_val_2[p]*0.7 for p in range(len_val)]
        sample_ub_2 = [(1-center_val_2[p])*0.3+center_val_2[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_2 = [max((random_x[p]-0.5)*2*(sample_lb_2[p]-center_val_2[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_2[p]-center_val_2[p]), 0) for p in range(len_val)]

        random_ratio_3 = [1-random_ratio_1[p]-random_ratio_2[p] for p in range(len_val)]
        beta[1][xi][t]= np.array([random_ratio_1,random_ratio_2, random_ratio_3])

proc[1] = {}
proc[1][16] = 79
proc[1][17] = 19
proc[1][18] = 102
proc[1][49] = 79
proc[1][48] = 50
proc[1][47] = 102
proc[1][76] = 50
proc[1][77] = 79
proc[1][78] = 19
proc[1][99] = 19
proc[1][100] = 102
proc[1][101] = 50
proc[1].update({c: c+1 for c in list(set(C[1])-set(I1[1])-set(I2[1])-set(I3[1])-set(I4[1]))})

pred[1] = {}
pred[1].update(dict([i, 15] for i in [16,17,18]))
pred[1].update(dict([i, 46] for i in [47,48,49]))
pred[1].update(dict([i, 75] for i in [76,77,78]))
pred[1].update(dict([i, 98] for i in [99,100,101]))
pred[1][19] = [17, 78, 99]
pred[1][50] = [48, 101, 76]
pred[1][79] = [49, 77, 16]
pred[1][102] = [100, 18, 47]
pred[1].update({c: c-1 for c in list(set(C[1])-set(I1[1])-set(I2[1])-set(I3[1])-set(I4[1])-set(M[1]))})

Jam_N[1] = np.zeros(len(C[1]))
for c in O[1]+D[1]: 
    Jam_N[1][c] = 99999
for c in C[1][0:15] + C[1][19:46] + C[1][50:66]:
    Jam_N[1][c] = 8
for c in C[1][67:75] + C[1][79:88] + C[1][90:98] + C[1][102:111]:
    Jam_N[1][c] = 4

for c in I1[1] + I2[1] + I3[1] + I4[1]:
    Jam_N[1][c] = 4
Jam_N[1][17] = 8
Jam_N[1][48] = 8
Jam_N[1][15] = 12
Jam_N[1][46] = 12
Jam_N[1][75] = 8
Jam_N[1][98] = 8

Q[1] = Jam_N[1]/4
Q[1][66] = 1
Q[1][89] = 1
for c in D[1]:
    Q[1][c] = 99999
""" Demand[1] = np.zeros(len(O[1]))
Demand[1][0] = 326
Demand[1][1] = 63 """
Demand[1] = [None]*len(O[1])
Demand[1][0] = [None]*num_scenario
Demand[1][1] = [None]*num_scenario
""" for xi in range(num_scenario):
    Demand[1][0][xi] = [None]*T
    Demand[1][1][xi] = [None]*T
    for t in range(T):
        Demand[1][0][xi][t] = 326
        Demand[1][1][xi][t] = 63 """
center_mean_1 = 326/1800
center_mean_2 = 63/1800
mean_ub_1 = center_mean_1*1.3
mean_lb_1 = center_mean_1*0.7
mean_ub_2 = center_mean_2*1.3
mean_lb_2 = center_mean_2*0.7
for xi in range(num_scenario):
    mean_1 = np.random.random()*(mean_ub_1-mean_lb_1) + mean_lb_1
    mean_2 = np.random.random()*(mean_ub_2-mean_ub_2) + mean_lb_2
    Demand[1][0][xi] = np.random.poisson(mean_1, T)
    Demand[1][1][xi] = np.random.poisson(mean_2, T)


C[2] = list(range(64))
O[2] = [52]
D[2] = [51]
BI[2] = [0, 21]
BO[2] = [20,41]
BI_IN[2] = [1, 3]
BO_IN[2] = [3, 1]
I1[2] = [13]
I2[2] = [14,27,28]
I3[2] = [62]
I4[2] = [63]
V[2] = [12,26,61]
M[2] = [15,29,42]
# beta[2] = np.array([[0.059,0,0.378],[0.941,0.969,0],[0,0.031,0.622]])
# beta[2] = np.array([[0.059, 0.969, 0.378], [0.941, 0.031, 0.622]])
beta[2] = [None]*num_scenario
for xi in range(num_scenario):
    beta[2][xi] = [None]*T
    for t in range(T):
        center_val_1 = [0.059, 0.969, 0.378]
        len_val = len(center_val_1)
        sample_lb_1 = [center_val_1[p]*0.7 for p in range(len_val)]
        sample_ub_1 = [(1-center_val_1[p])*0.3+center_val_1[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_1 = [max((random_x[p]-0.5)*2*(sample_lb_1[p]-center_val_1[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_1[p]-center_val_1[p]), 0) for p in range(len_val)]
        random_ratio_2 = [1-random_ratio_1[p] for p in range(len_val)]
        beta[2][xi][t] = np.array([random_ratio_1, random_ratio_2])
        # beta[2][xi][t] = np.array([[0.059, 0.969, 0.378], [0.941, 0.031, 0.622]])
proc[2]={}
proc[2][13] = 42
proc[2][14] = 15
proc[2][27] = 29
proc[2][28] = 42
proc[2][62] = 15
proc[2][63] = 29
proc[2].update({c: c+1 for c in list(set(C[2])-set(I1[2])-set(I2[2])-set(I3[2])-set(I4[2]))})
pred[2] = {}
pred[2][15] = [14,62]
pred[2][29] = [27,63]
pred[2][42] = [13, 28]
pred[2].update(dict([i, 12] for i in [13,14]))
pred[2].update(dict([i, 26] for i in [27,28]))
pred[2].update(dict([i, 61] for i in [62,63]))
pred[2].update({c: c-1 for c in list(set(C[2])-set(I1[2])-set(I2[2])-set(I3[2])-set(I4[2])-set(M[2]))})

Jam_N[2] = np.zeros(len(C[2]))
for c in O[2]+D[2]: 
    Jam_N[2][c] = 99999
for c in C[2][0:12] + C[2][15:26] + C[2][29:42]:
    Jam_N[2][c] = 8
for c in C[2][42:51] + C[2][53:61]:
    Jam_N[2][c] = 4

for c in I1[2] + I2[2] + I3[2] + I4[2]:
    Jam_N[2][c] = 4
Jam_N[2][14] = 8
Jam_N[2][27] = 8
Jam_N[2][12] = 12
Jam_N[2][26] = 12
Jam_N[2][61] = 8

Q[2] = Jam_N[2]/4
Q[2][52] = 1
for c in D[2]:
    Q[2][c] = 99999
""" Demand[2] = np.zeros(len(O[2]))
Demand[2][0] = 180 """
Demand[2] = [None]*len(O[2])
Demand[2][0] = [None]*num_scenario
""" for xi in range(num_scenario):
    Demand[2][0][xi] = [None]*T
    for t in range(T):
        Demand[2][0][xi][t] = 180 """
center_mean_1 = 180/1800
mean_ub_1 = center_mean_1*1.3
mean_lb_1 = center_mean_1*0.7
for xi in range(num_scenario):
    mean_1 = np.random.random()*(mean_ub_1-mean_lb_1) + mean_lb_1
    Demand[2][0][xi] = np.random.poisson(mean_1, T)



C[3] = list(range(72))
O[3] = [26,49]
D[3] = [48,71]
BI[3] = [0,13]
BO[3] = [12,25]
BI_IN[3] = [2, 4]
BO_IN[3] = [4, 2]
I1[3] = [6,17]
I2[3] = [7,8,18,19]
I3[3] = [36,59]
I4[3] = [37,38,60,61]
V[3] = [5,16,35,58]
M[3] = [9,20,39,62]
# [3] = np.array([[0.193,0.016,0.448,0.423],[0.774,0.88,0.362,0.051],[0.033,0.104,0.19,0.526]])
beta[3] = [None]*num_scenario
for xi in range(num_scenario):
    beta[3][xi] = [None]*T
    for t in range(T):
        center_val_1 = [0.193, 0.016, 0.448, 0.423] 
        len_val = len(center_val_1)
        sample_lb_1 = [center_val_1[p]*0.7 for p in range(len_val)]
        sample_ub_1 = [(1-center_val_1[p])*0.3+center_val_1[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_1 = [max((random_x[p]-0.5)*2*(sample_lb_1[p]-center_val_1[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_1[p]-center_val_1[p]), 0) for p in range(len_val)]

        center_val_2 = [0.774, 0.88, 0.362, 0.051]
        len_val = len(center_val_2)
        sample_lb_2 = [center_val_2[p]*0.7 for p in range(len_val)]
        sample_ub_2 = [(1-center_val_2[p])*0.3+center_val_2[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_2 = [max((random_x[p]-0.5)*2*(sample_lb_2[p]-center_val_2[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_2[p]-center_val_2[p]), 0) for p in range(len_val)]

        random_ratio_3 = [1-random_ratio_1[p]-random_ratio_2[p] for p in range(len_val)]
        beta[3][xi][t]= np.array([random_ratio_1,random_ratio_2, random_ratio_3])
        # beta[3][xi][t] = np.array([[0.193,0.016,0.448,0.423],[0.774,0.88,0.362,0.051],[0.033,0.104,0.19,0.526]])
proc[3]= {}
proc[3][6] = 39
proc[3][7] = 9
proc[3][8] = 62
proc[3][17] = 62
proc[3][18] = 20
proc[3][19] = 39
proc[3][36] = 20
proc[3][37] = 39
proc[3][38] = 9
proc[3][59] = 9
proc[3][60] = 62
proc[3][61] = 20
proc[3].update({c: c+1 for c in list(set(C[3])-set(I1[3])-set(I2[3])-set(I3[3])-set(I4[3]))})
pred[3] = {}
pred[3][9] = [7,38,59]
pred[3][20] = [18,61,36]
pred[3][39] = [37,19,6]
pred[3][62] = [8,60,17]
pred[3].update(dict([i, 5] for i in [6,7,8]))
pred[3].update(dict([i, 16] for i in [17,18,19]))
pred[3].update(dict([i, 35] for i in [36,37,38]))
pred[3].update(dict([i, 58] for i in [59,60,61]))
pred[3].update({c: c-1 for c in list(set(C[3])-set(I1[3])-set(I2[3])-set(I3[3])-set(I4[3])-set(M[3]))})

Jam_N[3] = np.zeros(len(C[3]))
for c in O[3]+D[3]: 
    Jam_N[3][c] = 99999
for c in C[3][0:5] + C[3][9:16] + C[3][20:26]:
    Jam_N[3][c] = 8
for c in C[3][27:35] + C[3][39:48] + C[3][50:58] + C[3][62:71]:
    Jam_N[3][c] = 4

for c in I1[3] + I2[3] + I3[3] + I4[3]:
    Jam_N[3][c] = 4
Jam_N[3][7] = 8
Jam_N[3][18] = 8
Jam_N[3][5] = 12
Jam_N[3][16] = 12
Jam_N[3][35] = 8
Jam_N[3][58] = 8

Q[3] = Jam_N[3]/4
Q[3][26] = 1
Q[3][49] = 1
for c in D[3]:
    Q[3][c] = 99999
""" Demand[3] = np.zeros(len(O[3]))
Demand[3][0] = 174
Demand[3][1] = 390 """
Demand[3] = [None]*len(O[3])
Demand[3][0] = [None]*num_scenario
Demand[3][1] = [None]*num_scenario
""" for xi in range(num_scenario):
    Demand[3][0][xi] = [None]*T
    Demand[3][1][xi] = [None]*T
    for t in range(T):
        Demand[3][0][xi][t] = 174
        Demand[3][1][xi][t] = 390 """
center_mean_1 = 174/1800
center_mean_2 = 390/1800
mean_ub_1 = center_mean_1*1.3
mean_lb_1 = center_mean_1*0.7
mean_ub_2 = center_mean_2*1.3
mean_lb_2 = center_mean_2*0.7
for xi in range(num_scenario):
    mean_1 = np.random.random()*(mean_ub_1-mean_lb_1) + mean_lb_1
    mean_2 = np.random.random()*(mean_ub_2-mean_ub_2) + mean_lb_2
    Demand[3][0][xi] = np.random.poisson(mean_1, T)
    Demand[3][1][xi] = np.random.poisson(mean_2, T)




C[4] = list(range(90))
O[4] = [44,67]
D[4] = [66,89]
BI[4] = [0,22]
BO[4] = [21,43]
BI_IN[4] = [3,5]
BO_IN[4] = [5,3]
I1[4] = [3,38]
I2[4] = [4,5,39,40]
I3[4] = [54,77]
I4[4] = [55,56,78,79]
V[4] = [2,37,53,76]
M[4] = [6,41,57,80]
# beta[4] = np.array([[0.012,0.189,0.28,0.444],[0.812,0.711,0.343,0.526],[0.176,0.1,0.377,0.03]])
beta[4] = [None]*num_scenario
for xi in range(num_scenario):
    beta[4][xi] = [None]*T
    for t in range(T):
        center_val_1 = [0.012, 0.189, 0.28, 0.444] 
        len_val = len(center_val_1)
        sample_lb_1 = [center_val_1[p]*0.7 for p in range(len_val)]
        sample_ub_1 = [(1-center_val_1[p])*0.3+center_val_1[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_1 = [max((random_x[p]-0.5)*2*(sample_lb_1[p]-center_val_1[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_1[p]-center_val_1[p]), 0) for p in range(len_val)]

        center_val_2 = [0.812, 0.711, 0.343, 0.526]
        len_val = len(center_val_2)
        sample_lb_2 = [center_val_2[p]*0.7 for p in range(len_val)]
        sample_ub_2 = [(1-center_val_2[p])*0.3+center_val_2[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_2 = [max((random_x[p]-0.5)*2*(sample_lb_2[p]-center_val_2[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_2[p]-center_val_2[p]), 0) for p in range(len_val)]

        random_ratio_3 = [1-random_ratio_1[p]-random_ratio_2[p] for p in range(len_val)]
        beta[4][xi][t]= np.array([random_ratio_1,random_ratio_2, random_ratio_3])
        # beta[4][xi][t] = np.array([[0.012,0.189,0.28,0.444],[0.812,0.711,0.343,0.526],[0.176,0.1,0.377,0.03]])
proc[4] = {}
proc[4].update(dict([i, 6] for i in [4,56,77]))
proc[4].update(dict([i, 41] for i in [39,79,54]))
proc[4].update(dict([i, 57] for i in [55,40,3]))
proc[4].update(dict([i, 80] for i in [78,5,38]))
proc[4].update({c: c+1 for c in list(set(C[4])-set(I1[4])-set(I2[4])-set(I3[4])-set(I4[4]))})
pred[4] = {}
pred[4][6] = [4,56,77]
pred[4][41] = [39,79,54]
pred[4][57] = [55,40,3]
pred[4][80] = [78,5,38]
pred[4].update(dict([i, 2] for i in [3,4,5]))
pred[4].update(dict([i, 37] for i in [38,39,40]))
pred[4].update(dict([i, 53] for i in [54,55,56]))
pred[4].update(dict([i, 76] for i in [77,78,79]))
pred[4].update({c: c-1 for c in list(set(C[4])-set(I1[4])-set(I2[4])-set(I3[4])-set(I4[4])-set(M[4]))})

Jam_N[4] = np.zeros(len(C[4]))
for c in O[4]+D[4]: 
    Jam_N[4][c] = 99999
for c in C[4][0:2] + C[4][6:37] + C[4][41:44]:
    Jam_N[4][c] = 8
for c in C[4][45:53] + C[4][57:66] + C[4][68:76] + C[4][80:89]:
    Jam_N[4][c] = 8

for c in I1[4] + I2[4] + I3[4] + I4[4]:
    Jam_N[4][c] = 4
Jam_N[4][4] = 8
Jam_N[4][39] = 8
Jam_N[4][55] = 8
Jam_N[4][78] = 8
Jam_N[4][2] = 12
Jam_N[4][37] = 12
Jam_N[4][53] = 12
Jam_N[4][76] = 12

Q[4] = Jam_N[4]/4
Q[4][44] = 2
Q[4][67] = 2
for c in D[4]:
    Q[4][c] = 99999
""" Demand[4] = np.zeros(len(O[4]))
Demand[4][0] = 897
Demand[4][1] = 432 """
Demand[4] = [None]*len(O[4])
Demand[4][0] = [None]*num_scenario
Demand[4][1] = [None]*num_scenario
""" for xi in range(num_scenario):
    Demand[4][0][xi] = [None]*T
    Demand[4][1][xi] = [None]*T
    for t in range(T):
        Demand[4][0][xi][t] = 897
        Demand[4][1][xi][t] = 432 """
center_mean_1 = 897/1800
center_mean_2 = 432/1800
mean_ub_1 = center_mean_1*1.3
mean_lb_1 = center_mean_1*0.7
mean_ub_2 = center_mean_2*1.3
mean_lb_2 = center_mean_2*0.7
for xi in range(num_scenario):
    mean_1 = np.random.random()*(mean_ub_1-mean_lb_1) + mean_lb_1
    mean_2 = np.random.random()*(mean_ub_2-mean_ub_2) + mean_lb_2
    Demand[4][0][xi] = np.random.poisson(mean_1, T)
    Demand[4][1][xi] = np.random.poisson(mean_2, T)



C[5] = list(range(104))
O[5] = [29,58,81]
D[5] = [28,80,103]
BI[5] = [0]
BO[5] = [57]
BI_IN[5] = [4]
BO_IN[5] = [4]
I1[5] = [16,39]
I2[5] = [17,18,40,41]
I3[5] = [68,91]
I4[5] = [69,70,92,93]
V[5] = [15,38,67,90]
M[5] = [19,42,71,94]   
# beta[5] = np.array([[0.05,0.068,0.226,0.717],[0.905,0.708,0.252,0.15],[0.045,0.224,0.522,0.133]])
beta[5] = [None]*num_scenario
for xi in range(num_scenario):
    beta[5][xi] = [None]*T
    for t in range(T):
        center_val_1 = [0.05, 0.068, 0.226, 0.717] 
        len_val = len(center_val_1)
        sample_lb_1 = [center_val_1[p]*0.7 for p in range(len_val)]
        sample_ub_1 = [(1-center_val_1[p])*0.3+center_val_1[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_1 = [max((random_x[p]-0.5)*2*(sample_lb_1[p]-center_val_1[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_1[p]-center_val_1[p]), 0) for p in range(len_val)]

        center_val_2 = [0.905, 0.708, 0.252, 0.15]
        len_val = len(center_val_2)
        sample_lb_2 = [center_val_2[p]*0.7 for p in range(len_val)]
        sample_ub_2 = [(1-center_val_2[p])*0.3+center_val_2[p] for p in range(len_val)]
        random_x = np.random.random(len_val)
        random_ratio_2 = [max((random_x[p]-0.5)*2*(sample_lb_2[p]-center_val_2[p]), 0)
            + max((random_x[p]-0.5)*2*(sample_ub_2[p]-center_val_2[p]), 0) for p in range(len_val)]

        random_ratio_3 = [1-random_ratio_1[p]-random_ratio_2[p] for p in range(len_val)]
        beta[5][xi][t]= np.array([random_ratio_1,random_ratio_2, random_ratio_3])
        # beta[5][xi][t] = np.array([[0.05,0.068,0.226,0.717],[0.905,0.708,0.252,0.15],[0.045,0.224,0.522,0.133]])
proc[5] = {}
proc[5].update(dict([i, 19] for i in [17,70,91]))
proc[5].update(dict([i, 42] for i in [40,93,68]))
proc[5].update(dict([i, 71] for i in [69,41,16]))
proc[5].update(dict([i, 94] for i in [92,18,39]))
proc[5].update({c: c+1 for c in list(set(C[5])-set(I1[5])-set(I2[5])-set(I3[5])-set(I4[5]))})
pred[5] = {}
pred[5][19] = [17,70,91]
pred[5][42] = [40,93,68]
pred[5][71] = [69,41,16]
pred[5][94] = [92,18,39]
pred[5].update(dict([i, 15] for i in [16,17,18]))
pred[5].update(dict([i, 38] for i in [39,40,41]))
pred[5].update(dict([i, 67] for i in [68,69,70]))
pred[5].update(dict([i, 90] for i in [91,92,93]))
pred[5].update({c: c-1 for c in list(set(C[5])-set(I1[5])-set(I2[5])-set(I3[5])-set(I4[5])-set(M[5]))})

Jam_N[5] = np.zeros(len(C[5]))
for c in O[5]+D[5]: 
    Jam_N[5][c] = 99999
for c in C[5][0:15] + C[5][30:38] + C[5][42:58]:
    Jam_N[5][c] = 8
for c in C[5][19:28]:
    Jam_N[5][c] = 12
for c in C[5][59:67]:
    Jam_N[5][c] = 4
for c in C[5][71:80] + C[5][82:90] + C[5][94:103]:
    Jam_N[5][c] = 8

for c in I1[5] + I2[5] + I3[5] + I4[5]:
    Jam_N[5][c] = 4
Jam_N[5][17] = 12
Jam_N[5][40] = 8
Jam_N[5][91] = 8
Jam_N[5][15] = 16
Jam_N[5][38] = 12
Jam_N[5][67] = 8
Jam_N[5][90] = 12

Q[5] = Jam_N[5]/4
Q[5][29] = 2
Q[5][58] = 1
Q[5][81] = 1
for c in D[5]:
    Q[5][c] = 99999
""" Demand[5] = np.zeros(len(O[5]))
Demand[5][0] = 1112
Demand[5][1] = 699
Demand[5][2] = 765 """
Demand[5] = [None]*len(O[5])
Demand[5][0] = [None]*num_scenario
Demand[5][1] = [None]*num_scenario
Demand[5][2] = [None]*num_scenario
""" for xi in range(num_scenario):
    Demand[5][0][xi] = [None]*T
    Demand[5][1][xi] = [None]*T
    Demand[5][2][xi] = [None]*T
    for t in range(T):
        Demand[5][0][xi][t] = 1112
        Demand[5][1][xi][t] = 699
        Demand[5][2][xi][t] = 765 """
center_mean_1 = 1112/1800
center_mean_2 = 699/1800
center_mean_3 = 765/1800
mean_ub_1 = center_mean_1*1.3
mean_lb_1 = center_mean_1*0.7
mean_ub_2 = center_mean_2*1.3
mean_lb_2 = center_mean_2*0.7
mean_ub_3 = center_mean_3*1.3
mean_lb_3 = center_mean_3*0.7
for xi in range(num_scenario):
    mean_1 = np.random.random()*(mean_ub_1-mean_lb_1) + mean_lb_1
    mean_2 = np.random.random()*(mean_ub_2-mean_ub_2) + mean_lb_2
    mean_3 = np.random.random()*(mean_ub_3-mean_ub_3) + mean_lb_3
    Demand[5][0][xi] = np.random.poisson(mean_1, T)
    Demand[5][1][xi] = np.random.poisson(mean_2, T)
    Demand[5][2][xi] = np.random.poisson(mean_3, T)

# print(beta)
# test data
""" for i in range(N):
    for xi in range(num_scenario):
        for t in range(T):
            # print(beta[i][xi][t])
            my_list = beta[i][xi][t][beta[i][xi][t] <= 0]
            if my_list != []:
                print("wrong")
for i in range(N):
    for xi in range(num_scenario):
        for c in range(len(O[i])):
            print(Demand[i][c][xi]) """
""" for i in range(N):
    for j in range(len(Demand[i])):
        for xi in range(num_scenario):
            for t in range(T):
                Demand[i][j][xi][t] = Demand[i][j][xi][t]/1800 """

n_init = [None]*N
for i in range(N):
    n_init[i] = np.ones(len(C[i]))*0.5
W=1/3
alpha=0.1/T

add=np.zeros(N)
add[0]=0
add[1]=80
add[2]=192
add[3]=256
add[4]=328
add[5]=418



C_ALL = list(range(522))
O_ALL = O[0]+ [d+80 for d in O[1]] + [d+192 for d in O[2]] + [d+256 for d in O[3]] + [d+ 328 for d in O[4]] + [d+418 for d in O[5]]
D_ALL = D[0]+ [d+80 for d in D[1]] + [d+192 for d in D[2]] + [d+256 for d in D[3]] + [d+ 328 for d in D[4]] + [d+418 for d in D[5]]
V_ALL = V[0]+ [d+80 for d in V[1]] + [d+192 for d in V[2]] + [d+256 for d in V[3]] + [d+ 328 for d in V[4]] + [d+418 for d in V[5]]
M_ALL = M[0]+ [d+80 for d in M[1]] + [d+192 for d in M[2]] + [d+256 for d in M[3]] + [d+ 328 for d in M[4]] + [d+418 for d in M[5]]
BI_ALL = BI[0]+ [d+80 for d in BI[1]] + [d+192 for d in BI[2]] + [d+256 for d in BI[3]] + [d+ 328 for d in BI[4]] + [d+418 for d in BI[5]]
BO_ALL = BO[0]+ [d+80 for d in BO[1]] + [d+192 for d in BO[2]] + [d+256 for d in BO[3]] + [d+ 328 for d in BO[4]] + [d+418 for d in BO[5]]
I1_ALL = I1[0]+ [d+80 for d in I1[1]] + [d+192 for d in I1[2]] + [d+256 for d in I1[3]] + [d+ 328 for d in I1[4]] + [d+418 for d in I1[5]]
I2_ALL = I2[0]+ [d+80 for d in I2[1]] + [d+192 for d in I2[2]] + [d+256 for d in I2[3]] + [d+ 328 for d in I2[4]] + [d+418 for d in I2[5]]
I3_ALL = I3[0]+ [d+80 for d in I3[1]] + [d+192 for d in I3[2]] + [d+256 for d in I3[3]] + [d+ 328 for d in I3[4]] + [d+418 for d in I3[5]]
I4_ALL = I4[0]+ [d+80 for d in I4[1]] + [d+192 for d in I4[2]] + [d+256 for d in I4[3]] + [d+ 328 for d in I4[4]] + [d+418 for d in I4[5]]

I = [None] * N
for i in range(N):
    I[i] = [None] * 4
    I[i][0] = [int(d+add[i]) for d in I1[i]]
    I[i][1] = [int(d+add[i]) for d in I2[i]]
    I[i][2] = [int(d+add[i]) for d in I3[i]]
    I[i][3] = [int(d+add[i]) for d in I4[i]]

# Demand = np.ones((len(O_ALL),T))

Demand_ALL = Demand[0]
Q_ALL = Q[0]
Jam_N_ALL = Jam_N[0]
n_init_all = n_init[0]
for i in range(1,N):
    Demand_ALL = Demand_ALL + [de for de in Demand[i]]
    Q_ALL = np.hstack((Q_ALL, Q[i]))
    Jam_N_ALL = np.hstack((Jam_N_ALL, Jam_N[i]))
    n_init_all = np.concatenate((n_init_all, n_init[i]), axis=0)

proc_all={}
proc_all.update(proc[0])
proc_all.update({k+80: proc[1][k]+80 for k in proc[1]})
proc_all.update({k+192: proc[2][k]+192 for k in proc[2]})
proc_all.update({k+256: proc[3][k]+256 for k in proc[3]})
proc_all.update({k+328: proc[4][k]+328 for k in proc[4]})
proc_all.update({k+418: proc[5][k]+418 for k in proc[5]})
proc_all[28] = 80
proc_all[32+80] = 192
proc_all[145] = 29
proc_all[212] = 256
proc_all[233] = 113
proc_all[268] = 328
proc_all[281] = 213
proc_all[349] = 418
proc_all[371] = 269
proc_all[418+57] = 328+22

pred_all={}
pred_all.update(pred[0])
for  key,value in pred[1].items():
    if isinstance(value,int):
        pred_all.update({key+80: value+80})
    else:
        pred_all.update({key+80: [v+80 for v in value]})
for  key,value in pred[2].items():
    if isinstance(value,int):
        pred_all.update({key+192: value+192})
    else:
        pred_all.update({key+192: [v+192 for v in value]})
for  key,value in pred[3].items():
    if isinstance(value,int):
        pred_all.update({key+256: value+256})
    else:
        pred_all.update({key+256: [v+256 for v in value]})
for  key,value in pred[4].items():
    if isinstance(value,int):
        pred_all.update({key+328: value+328})
    else:
        pred_all.update({key+328: [v+328 for v in value]})
for  key,value in pred[5].items():
    if isinstance(value,int):
        pred_all.update({key+418: value+418})
    else:
        pred_all.update({key+418: [v+418 for v in value]})
pred_all[29] = 65+80
pred_all[0+80] = 28
pred_all[33+80] = 41+192
pred_all[192] = 32+80
pred_all[21+192] = 25+256
pred_all[256] = 20+192
pred_all[13+256] = 43+328
pred_all[328] = 12+256
pred_all[22+328] = 57+418
pred_all[418] = 328+21