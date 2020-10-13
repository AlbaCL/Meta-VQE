############################################################################
#   Main code meta-VQE single-transmon simulation
#   ______________________________________________
#
# Self-contained (all functions are defined in this script)
# Structure:
# 1. Preamble: all information to run the simulations
#       optimization method, model, number of encoding & processing layer, paths...
# 2. Hamiltonian: hamiltonian function and exact diagonalization function
# 3. Ansatz: both the original QCAD ansatz and the one used in the meta-VQE paper
#       includes the encoding ansatz
# 4. Meta-VQE: test and training
# 5. Standard VQE
# 6. "Smart" VQE (VQE with encoding layer)
# 7. opt-meta-VQE
# 8. Plots

import tequila as tq
from tequila.hamiltonian.paulis import X,Y,Z,I
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os 
import time

seeds = ['matterlab', 'alba', 'jakob', 'alan']

s = 0
# Seed generator from alphabetic string
def seed_gen(string):
    num_str = [ord(letter) - 96 for letter in string]
    # seed must be between 0 and 2**32 - 1
    return(np.mod(int(''.join(map(str,num_str))),2**32 - 1))

random.seed(seed_gen(seeds[s])) # default seed

model_name = 'transmon'
n_qub = 4 # qubits
n_lay = [1,1] # layers [encoding, variational circuit]
n_train = 10 # training points for the metaVQE
n_test = 50 # test " "

# Hamiltonian arguments: flux
arg_max = 0.0
arg_min = 0.5

equispaced_train = True
equispaced_test = True
test_dif_train = True # test points different to training meta-VQE points (shifted)
                      # CAREFUL: they are currently fixed to 10 points [0.0, 0.5]
                      # search "test_dif_train" in the code

# Extra tests
test_VQE = True # standard VQE for each training point (same # layers wrt metaVQE)
test_VQE_smart = True # VQE with encoding and using previous point as initiaization
test_opt_metaVQE = True # standard VQE with metaVQE solution as initialization

# Optimizer options
methods = 'BFGS'
grad_methods = '2-point'
backend = 'qulacs'
lr = 0.01
mthd_opt = {'finite_diff_rel_step': 0.0001, 'maxiter':200}

# Paths to save data and figures

# extra message: 
# in case you want to save two figures with the same n_qub, n_lay, ans, model
# original: original QCAD ansatz
original = False
extra_mes = ""

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data/{}/nqub_{}/nlay_{}'.format(model_name, n_qub, n_lay))
plot_dir = os.path.join(script_dir, 'img/{}/'.format(model_name))

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

# print all info
file_name = os.path.join(data_dir, "basic_data_seed_{}{}.txt".format(seeds[s],extra_mes))
file_data = open(file_name, "w")

path = file_data
print(model_name)
print("qubits = ", n_qub, file=path)
print("layers [encoding, variational circuit] = ", n_lay, file=path)
print("training points = ", n_train, file=path)
print("test points = ", n_test, file=path)
print("equispaced train and test = ", equispaced_train, ", ", equispaced_test, file=path)
print("Optimization options: ", methods, ", ", grad_methods, ", ", backend, ", lr = ", lr, ", other: ",mthd_opt, file=path)

file_data.close()

# Variable counting
file_varcount = os.path.join(data_dir, "variable_counting_seed_{}{}.txt".format(seeds[s],extra_mes))
file_data_varcount = open(file_varcount, "w")
print("Variable counting: number of optimization variables", file=file_data_varcount)

######################
# HAMILTONIAN
######################

def Htrans(f):
    N = -4.0*Z(3) - 2.0*Z([2,3]) - Z([1,2,3]) - 0.5*Z([0,1,2,3]) - 0.5*I()
    cosine = 0.5*X(0) + 0.25*X(1)- 0.25*Z(0)*X(1)
    cosine += 0.125*(X(2) - Z(1)*X(2) + Z(0)*X(2) - Z([0,1])*X(2))
    cosine += 0.0625*(X(3) - Z(2)*X(3) + Z(1)*X(3) - Z([1,2])*X(3) + Z(0)*X(3) - Z([0,2])*X(3) + Z([0,1])*X(3) -Z([0,1,2])*X(3))

    Csum = 91
    e = 1.60217662 * 10**(-19) #elementary charge
    h = 6.62607004 * 10**(-34) #Planck constant
    Ec = 0.2129 #e**2/(2*Csum*1e-15)/h*1e-9
    Ej = 20 
    Ejt = Ej*np.abs(np.cos(2*np.pi*f))

    ham = 4*Ec*N*N - 2*Ejt*cosine

    return(ham)

# Exact diagonalization
def Htrans_exact(ham):
    ham_matrix = ham.to_matrix()
    energ = np.linalg.eigvals(ham_matrix)
    return(min(energ).real)

#############################
# ANSATZ
#############################

def XX(q0,q1,angle):
    return(tq.gates.ExpPauli(paulistring="X({})X({})".format(q0,q1), angle=tq.Variable(name=angle)))

if original==False:
    def ans4(num_qubits, layers, arg, enc):
        U = tq.QCircuit()
        for l in range(layers):
            # Layer single-qubit gates
            for q in range(num_qubits):
                # No encoding:
                if enc == False:
                    theta = tq.Variable(name="th_{}{}".format(l, q))
                    phi = tq.Variable(name="ph_{}{}".format(l, q))
                    U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)
                # Encoding:
                if enc == True:
                    theta = tq.Variable(name="thenc_{}{}".format(l, q))
                    phi = tq.Variable(name="phenc_{}{}".format(l, q))
                    wth = tq.Variable(name="thw_{}{}".format(l, q))
                    wph = tq.Variable(name="phw_{}{}".format(l, q))
                    U += tq.gates.Rx(target=q, angle=wth*arg + theta) +  tq.gates.Rz(target=q, angle=wph*arg + phi)
                # Layer XX gates      
            # no encoding in entangling gates
            # SAME entangling gates
            U += XX(0,1,"a") + XX(2,3,"b") + XX(1,2,"c") + XX(0,3,"d") + XX(0,2,"e") + XX(1,3,"f")
            #if enc == False:
                #U += XX(0,1, "a{}".format(l)) + XX(2,3,"b{}".format(l)) + XX(1,2,"c{}".format(l)) + XX(0,3,"d{}".format(l)) + XX(0,2,"e{}".format(l)) + XX(1,3,"f{}".format(l))
            #if enc == True:
            #   U += XX(0,1, "aw{}".format(l)) + XX(2,3,"bw{}".format(l)) + XX(1,2,"cw{}".format(l)) + XX(0,3,"dw{}".format(l)) + XX(0,2,"ew{}".format(l)) + XX(1,3,"fw{}".format(l))      
        return (U)
if original==True:
    def ans4(num_qubits, layers, arg, enc):
        U = tq.QCircuit()
        for l in range(layers):
            # Layer single-qubit gates
            for q in range(num_qubits):
                # No encoding:
                if enc == False:
                    theta = tq.Variable(name="th_{}{}".format(l, q))
                    phi = tq.Variable(name="ph_{}{}".format(l, q))
                    U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)
                # Encoding:
                if enc == True:
                    theta = tq.Variable(name="thenc_{}{}".format(l, q))
                    phi = tq.Variable(name="phenc_{}{}".format(l, q))
                    wth = tq.Variable(name="thw_{}{}".format(l, q))
                    wph = tq.Variable(name="phw_{}{}".format(l, q))
                    U += tq.gates.Rx(target=q, angle=wth*arg + theta) +  tq.gates.Rz(target=q, angle=wph*arg + phi)
            # Layer XX gates      
            # no encoding in entangling gates
            # SAME entangling gates
            if enc == False:
                U += XX(0,1, "a{}".format(l)) + XX(2,3,"b{}".format(l)) + XX(1,2,"c{}".format(l)) + XX(0,3,"d{}".format(l)) + XX(0,2,"e{}".format(l)) + XX(1,3,"f{}".format(l))
            if enc == True:
                U += XX(0,1, "aw{}".format(l)) + XX(2,3,"bw{}".format(l)) + XX(1,2,"cw{}".format(l)) + XX(0,3,"dw{}".format(l)) + XX(0,2,"ew{}".format(l)) + XX(1,3,"fw{}".format(l))      
            
            # second layer single-qubit gates
            for q in range(num_qubits):
                # No encoding:
                if enc == False:
                    theta = tq.Variable(name="th2_{}{}".format(l, q))
                    phi = tq.Variable(name="ph2_{}{}".format(l, q))
                    U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)
                # Encoding:
                if enc == True:
                    theta = tq.Variable(name="thenc2_{}{}".format(l, q))
                    phi = tq.Variable(name="phenc2_{}{}".format(l, q))
                    wth = tq.Variable(name="thw2_{}{}".format(l, q))
                    wph = tq.Variable(name="phw2_{}{}".format(l, q))
                    U += tq.gates.Rx(target=q, angle=wth*arg + theta) +  tq.gates.Rz(target=q, angle=wph*arg + phi)   
        return (U)

#######################
#    meta-VQE
#######################

# TRAINING

t0_metaVQE = time.time()

print("MetaVQE training running")

# Generate the training points
if equispaced_train == True:
    arg_train = [arg_min + i*(arg_max-arg_min)/(n_train-1) for i in range(n_train)]
else:
    arg_train = [random.uniform(arg_min, arg_max) for i in range(n_train)]
    arg_train.sort()

Obj = tq.Objective()
for i in range(n_train):
    Ham = Htrans(arg_train[i])
    total_U = ans4(n_qub, n_lay[0], arg_train[i], True) + ans4(n_qub, n_lay[1], arg_train[i], False)

    if original == False:
        # last layer of single-qubit gates    
        for q in range(n_qub):
            theta = tq.Variable(name="th2_{}".format(q))
            phi = tq.Variable(name="ph2_{}".format(q))
            # No encoding:
            total_U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)

    Obj += tq.ExpectationValue(H=Ham, U=total_U)

# Number of optimization variables
print("Meta-VQE training: ", len(Obj.extract_variables()), file=file_data_varcount)

variables = Obj.extract_variables()
variables = sorted(variables, key=lambda x: x.name)

# Random initialization of variables
th0 = {key: random.uniform(0, np.pi) for key in variables}

initial_values = th0
metaVQE = tq.minimize(objective=Obj, adaptive = True, lr=lr, method_options=mthd_opt, method=methods, gradient=grad_methods, samples=None,
                      initial_values=initial_values, backend=backend, noise=None, device=None, silent=True)

file_name = os.path.join(data_dir, "metaVQE_seed_{}{}.txt".format(seeds[s], extra_mes))
file_data = open(file_name, "w")
print(metaVQE, file=file_data)
file_data.close()

# Results training
x_train = arg_train
y_train_ex = []
y_train_metaVQE = []
error_train_metaVQE = []
file_name = os.path.join(data_dir, "metaVQE_train_seed_{}{}.txt".format(seeds[s], extra_mes))
file_data = open(file_name, "w")

for i in range(n_train):
    Ham = Htrans(arg_train[i])
    total_U = ans4(n_qub, n_lay[0], arg_train[i], True) + ans4(n_qub, n_lay[1], arg_train[i], False)

    if original == False:
        # last layer of single-qubit gates    
        for q in range(n_qub):
            theta = tq.Variable(name="th2_{}".format(q))
            phi = tq.Variable(name="ph2_{}".format(q))
            # No encoding:
            total_U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)

    exp_val = tq.ExpectationValue(H=Ham, U=total_U)
    res = tq.simulate(exp_val, variables=metaVQE.angles)
    res_ex = Htrans_exact(Ham)

    print(arg_train[i], res_ex, res, abs(res-res_ex), file=file_data)

    y_train_ex.append(res_ex)
    y_train_metaVQE.append(res)
    error_train_metaVQE.append(abs(res-res_ex))

file_data.close()

# TEST

print("MetaVQE test running")

# Results test

# Generate the training points
if equispaced_test == True:
    arg_test = [arg_min + i*(arg_max-arg_min)/(n_test-1) for i in range(n_test)]
    arg_test.sort()
else:
    arg_test = [random.uniform(arg_min, arg_max) for i in range(n_test)]
    arg_test.sort()

x_test = arg_test
y_test_ex = []
y_test_metaVQE = []
error_test_metaVQE = []
file_name = os.path.join(data_dir, "metaVQE_test_seed_{}{}.txt".format(seeds[s], extra_mes))
file_data = open(file_name, "w")

Obj = tq.Objective()
for i in range(n_test):
    print(i, "testing")
    Ham = Htrans(arg_test[i])
    total_U = ans4(n_qub, n_lay[0], arg_test[i], True) + ans4(n_qub, n_lay[1], arg_test[i], False)

    if original == False:
        # last layer of single-qubit gates    
        for q in range(n_qub):
            theta = tq.Variable(name="th2_{}".format(q))
            phi = tq.Variable(name="ph2_{}".format(q))
            # No encoding:
            total_U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)

    exp_val = tq.ExpectationValue(H=Ham, U=total_U)
    res = tq.simulate(exp_val, variables=metaVQE.angles)
    res_ex = Htrans_exact(Ham)

    print(arg_test[i], res_ex, res, abs(res - res_ex), file=file_data)

    y_test_metaVQE.append(res)
    y_test_ex.append(res_ex)
    error_test_metaVQE.append(abs(res-res_ex))

file_data.close()

print("Meta-VQE test: ", len(total_U.extract_variables()), file=file_data_varcount)

t1_metaVQE = time.time()

t_metaVQE = abs(t1_metaVQE-t0_metaVQE)

#############################
# TEST and TRAIN different
#############################

if test_dif_train == True:
    arg_train = [0.0 + i*0.05 for i in range(10)]

#######################
# STANDARD VQE
# For meta-VQE training points only
# initial parameters the result of the previous point minimization
#######################

t0_VQE = time.time()

if test_VQE == True:

    print("standard VQE test running")

    y_stVQE = []
    error_stVQE = []
    file_name1 = os.path.join(data_dir, "VQE_seed_{}{}.txt".format(seeds[s], extra_mes))
    file_name2 = os.path.join(data_dir, "VQE_test_seed_{}{}.txt".format(seeds[s], extra_mes))
    file_data1 = open(file_name1, "w")
    file_data2 = open(file_name2, "w")

    init_var = {key: random.uniform(0, np.pi) for key in variables}

    for i in range(n_train):
        Ham = Htrans(arg_train[i])
        # same total number of layers, no encoding.
        total_U = ans4(n_qub, n_lay[0]+n_lay[1], arg_train[i], False)

        if original == False:
            # last layer of single-qubit gates    
            for q in range(n_qub):
                theta = tq.Variable(name="th2_{}".format(q))
                phi = tq.Variable(name="ph2_{}".format(q))
                # No encoding:
                total_U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)

        Obj = tq.ExpectationValue(H=Ham, U=total_U)

        variables = Obj.extract_variables()
        variables = sorted(variables, key=lambda x: x.name)

        initial_values = init_var
        standardVQE = tq.minimize(objective=Obj, adaptive = True, lr=lr, method_options=mthd_opt, method=methods, gradient=grad_methods, samples=None,
                                  initial_values=initial_values, backend=backend, noise=None,
                                  device=None, silent=True)
        res = standardVQE.energy
        if test_dif_train == True:
            res_ex = Htrans_exact(Ham)
        else:
            res_ex = y_train_ex[i] # already computed in metaVQE train

        init_var = standardVQE.angles

        print(standardVQE, file=file_data1)
        
        print(arg_train[i], res_ex, res, abs(res - res_ex), file=file_data2)

        print(i," test point trained")
   
        y_stVQE.append(res)
        error_stVQE.append(abs(res-res_ex))

    file_data1.close()
    file_data2.close()

    print("Standard VQE: ", len(Obj.extract_variables()), file=file_data_varcount)

t1_VQE = time.time()

t_VQE = abs(t1_VQE-t0_VQE)

#######################
# STANDARD VQE with encoding 
# and using the previous optimization 
# point as starting point + encoding
#######################

t0_VQE_smart = time.time()

if test_VQE_smart == True:

    print("standard VQE smart test running")

    y_stVQE_smart = []
    error_stVQE_smart = []
    file_name1 = os.path.join(data_dir, "VQE_smart_seed_{}{}.txt".format(seeds[s], extra_mes))
    file_name2 = os.path.join(data_dir, "VQE_smart_test_seed_{}{}.txt".format(seeds[s], extra_mes))
    file_data1 = open(file_name1, "w")
    file_data2 = open(file_name2, "w")

    # Random initialization of variables for the first point
    init_var = th0 # same as meta-VQE = {key: random.uniform(0, np.pi) for key in variables}

    for i in range(n_train):
        Ham = Htrans(arg_train[i])
        # same as meta-VQE.
        total_U = ans4(n_qub, n_lay[0], arg_train[i], True) + ans4(n_qub, n_lay[1], arg_train[i], False)

        # last layer of single-qubit gates    
        if original == False:
            # last layer of single-qubit gates    
            for q in range(n_qub):
                theta = tq.Variable(name="th2_{}".format(q))
                phi = tq.Variable(name="ph2_{}".format(q))
                # No encoding:
                total_U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)

        Obj = tq.ExpectationValue(H=Ham, U=total_U)

        variables = Obj.extract_variables()
        variables = sorted(variables, key=lambda x: x.name)

        initial_values = init_var
        VQE_smart = tq.minimize(objective=Obj, adaptive = True, lr=lr, method_options=mthd_opt, method=methods, gradient=grad_methods, samples=None,
                                  initial_values=initial_values, backend=backend, noise=None,
                                  device=None, silent=True)
        res = VQE_smart.energy

        if test_dif_train == True:
            res_ex = Htrans_exact(Ham)
        else:
            res_ex = y_train_ex[i] # already computed in metaVQE train

        print(VQE_smart, file=file_data1)

        # initialization for the next point
        init_var = VQE_smart.angles
        
        print(arg_train[i], res_ex, res, abs(res - res_ex), file=file_data2)

        print(i," test point trained")
   
        y_stVQE_smart.append(res)
        error_stVQE_smart.append(abs(res-res_ex))

    file_data1.close()
    file_data2.close()

    print("Standard VQE smart: ", len(Obj.extract_variables()), file=file_data_varcount)

t1_VQE_smart = time.time()

t_VQE_smart = abs(t1_VQE_smart-t0_VQE_smart)

#######################
# opt-meta-VQE
# (VQE with encoding and meta-VQE initialization)
#######################

t0_opt_metaVQE = time.time()

if test_opt_metaVQE == True:

    print("opt-meta-VQE test running")

    y_opt_metaVQE = []
    error_opt_metaVQE = []
    file_name1 = os.path.join(data_dir, "opt_metaVQE_seed_{}{}.txt".format(seeds[s], extra_mes))
    file_name2 = os.path.join(data_dir, "opt_metaVQE_test_seed_{}{}.txt".format(seeds[s], extra_mes))
    file_data1 = open(file_name1, "w")
    file_data2 = open(file_name2, "w")

    for i in range(n_train):
        Ham = Htrans(arg_train[i])
        # same as meta-VQE.
        total_U = ans4(n_qub, n_lay[0], arg_train[i], True) + ans4(n_qub, n_lay[1], arg_train[i], False)

        # last layer of single-qubit gates    
        if original == False:
        # last layer of single-qubit gates    
            for q in range(n_qub):
                theta = tq.Variable(name="th2_{}".format(q))
                phi = tq.Variable(name="ph2_{}".format(q))
                # No encoding:
                total_U += tq.gates.Rx(target=q, angle=theta) + tq.gates.Rz(target=q, angle=phi)

        Obj = tq.ExpectationValue(H=Ham, U=total_U)

        variables = Obj.extract_variables()
        variables = sorted(variables, key=lambda x: x.name)

        initial_values = metaVQE.angles
        opt_metaVQE = tq.minimize(objective=Obj, adaptive = True, lr=lr, method_options=mthd_opt, method=methods, gradient=grad_methods, samples=None,
                                  initial_values=initial_values, backend=backend, noise=None,
                                  device=None, silent=True)
        res = opt_metaVQE.energy

        if test_dif_train == True:
            res_ex = Htrans_exact(Ham)
        else:
            res_ex = y_train_ex[i] # already computed in metaVQE train

        print(opt_metaVQE, file=file_data1)
        
        print(arg_train[i], res_ex, res, abs(res - res_ex), file=file_data2)

        print(i," test point trained")
   
        y_opt_metaVQE.append(res)
        error_opt_metaVQE.append(abs(res-res_ex))

    file_data1.close()
    file_data2.close()

    print("opt-meta-VQE: ", len(Obj.extract_variables()), file=file_data_varcount)

    t1_opt_metaVQE = time.time()

    t_opt_metaVQE = abs(t1_opt_metaVQE-t0_opt_metaVQE)

########
# TIME
########

file_time = os.path.join(data_dir, "time_seed_{}{}.txt".format(seeds[s], extra_mes))
file_data_time = open(file_time, "w")

print("meta VQE (test+training): ", t_metaVQE/60," min.", file=file_data_time)
print("VQE (training points): ", t_VQE/60," min.", file=file_data_time)
print("VQE with encoding (training points): ", t_VQE_smart/60," min.", file=file_data_time)
print("opt-meta-VQE (training points): ", t_opt_metaVQE/60," min.", file=file_data_time)

file_data_time.close()

#####################
# PLOTS
#####################

# Energy
fig1 = plt.figure(0)

plt.plot(x_test, y_test_ex, color="black", ls=":", label="exact")
plt.plot(x_test, y_test_metaVQE,  ls="-", label="metaVQE test")
plt.scatter(x_train, y_train_metaVQE, color="red", marker="x", label="metaVQE train")
if test_VQE == True:
    plt.plot(x_train, y_stVQE, marker=".", label="VQE")
if test_VQE_smart == True:
    plt.plot(x_train, y_stVQE_smart, marker="x", label="VQE enc")
if test_opt_metaVQE == True:
    plt.plot(x_train, y_opt_metaVQE, marker="*", label="opt-meta-VQE")

plt.xlabel("f")
plt.ylabel('g.s. energy')
plt.legend()
fig_name = "{}_n{}_{}lay_seed_{}{}.png".format(model_name, n_qub, n_lay, seeds[s], extra_mes)

fig1.savefig(plot_dir + fig_name)

# Absolute error meta-VQE

fig1 = plt.figure(1)

plt.plot(x_test, error_test_metaVQE, ls="-", label="metaVQE test")
plt.scatter(x_train, error_train_metaVQE, marker="x", color="red", label="metaVQE train")

plt.xlabel('f')
plt.ylabel("Absolute error")
plt.legend()
fig_name_error = "{}_n{}_{}lay_seed_{}{}_error_metaVQE.png".format(model_name, n_qub, n_lay, seeds[s], extra_mes)

fig1.savefig(plot_dir + fig_name_error)

# Absolute error VQE vs opt-meta-VQE

fig1 = plt.figure(2)

#plt.plot(x_test, error_test_metaVQE, ls="-", label="metaVQE test")
#plt.scatter(x_train, error_train_metaVQE, marker="x", color="red", label="metaVQE train")
if test_VQE == True:
    plt.plot(x_train, error_stVQE, marker=".", label="VQE")
if test_VQE_smart == True:
    plt.plot(x_train, error_stVQE_smart, marker="x", label="VQE enc")
if test_opt_metaVQE == True:
    plt.plot(x_train, error_opt_metaVQE, marker="*", label="opt-meta-VQE")

plt.xlabel('f')
plt.ylabel("Absolute error")
plt.legend()
fig_name_error = "{}_n{}_{}lay_seed_{}{}_error.png".format(model_name, n_qub, n_lay, seeds[s], extra_mes)

fig1.savefig(plot_dir + fig_name_error)