import numpy as np
import matplotlib.pyplot as plt 
import time

x_axis = np.arange(-5,5,0.1)
y_axis = np.arange(-5,5,0.1)

xx_axis, yy_axis = np.meshgrid(x_axis, y_axis)

#Q13


def q(x):
	n = np.shape(x)
	(s_1, s_2, s_3) = (0,0,0)
	for i in range(n[0]):
		s_1 = s_1 + x[i]**2
		s_3 = s_3 + x[i]

	for i in range(n[0] - 1):
		s_2 = s_2 + x[i]*x[i+1]


	Q = s_1 - s_2 - s_3 
	return Q



def q_2(x_1,x_2):
	Q = x_1**2 + x_2**2 -x_1-x_2
	return Q 

q = q_2(xx_axis,yy_axis)
fig = plt.figure("Level map 3D")
ax3d = fig.gca(projection="3d")
surf = ax3d.plot_surface(xx_axis, yy_axis, q, cmap="inferno", edgecolor='none')
plt.title("Surface représentative de q pour n=2")
plt.colorbar(surf)
plt.show()

plt.figure("Level map 2D")
lm2D = plt.contour(x_axis, y_axis, q, 200, cmap="inferno")
plt.title("Lignes de niveau de q")
plt.colorbar(lm2D)
plt.show()

def HessFQ(x):
	n = np.shape(x)
	N = n[0]
	H = 2*np.eye(N)
	for i in range(N-1):
		H[i,i+1]=-1
		H[i+1,i]=-1
	return H


def gradFQ(x):
	n = np.shape(x)
	A = HessFQ(x)
	b = np.ones(n[0])
	b = np.transpose(b)
	G = A@x - b
	return G 





#Fortement convexe

x_axis_2 = np.arange(-3,3,0.1)
y_axis_2 = np.arange(-3,3,0.1)

xx_axis_2, yy_axis_2 = np.meshgrid(x_axis_2, y_axis_2)

def g(x_1,x_2):
	G = (x_1)**2+((x_2)**2)*((x_2)**2 +2)
	return G

def gradFC(x):
	G = np.zeros(2)
	G[0] = 2*x[0]
	G[1] = 4*x[1]*(x[1]**2 + 1)
	return G

def HessFC(x):
	H = np.zeros((2,2))
	H[0,0] = 2
	H[0,1] = 0
	H[1,0] = H[0,1]
	H[1,1] = 12*x[1]**2 + 4
	return H
#Q16

gtest = g(xx_axis_2,yy_axis_2)
fig2 = plt.figure("Level map 3D g(x)")
ax3D_2 = fig2.gca(projection="3d")
surf_2 = ax3D_2.plot_surface(xx_axis_2,yy_axis_2, gtest, cmap="hot", edgecolor='none')
plt.title("Surface représentative de g ")
plt.colorbar(surf_2)
plt.show()

plt.figure("Level map 2D de g")
lm2D_2 = plt.contour(x_axis_2,y_axis_2, gtest, 200, cmap ="hot")
plt.title("Lignes de niveaux de g")
plt.colorbar(lm2D_2)
plt.show()




#d/iii



#Rosembrok

def Rosembrok(x1,x2):
	Rsbrk = (x1-1)**2+ 10*((x1)**2 - x2)**2
	return Rsbrk

r = Rosembrok(xx_axis_2,yy_axis_2)
fig3 = plt.figure("Level map 3D Rosembrok")
ax3D_r = fig3.gca(projection="3d")
surf_3 = ax3D_r.plot_surface(xx_axis_2,yy_axis_2,r, cmap="plasma", edgecolor="none")
plt.title("Surface représentative de Rosembrok")
plt.colorbar(surf_3)
plt.show()

plt.figure("Level map 2D de Rosembrok")
lm2D_r = plt.contour(x_axis_2,y_axis_2, r,200, cmap="plasma")
plt.title("Lignes de niveaux de Rosembrok")
plt.colorbar(lm2D_r)
plt.show()


def evalFR(x):
	F_r = Rosembrok(x[0],x[1])
	return F_r

def gradFR(x):
	G = np.zeros(2)
	G[0]  = 2*x[0] - 2 + 40*((x[0])**3) - 40*x[1]*x[0]
	G[1] = -20*(x[0])**2 + 20*x[1]
	return G 
 
def hessFR(x):
	H = np.zeros((2,2))
	H[0,0]= 2 + 120*((x[0])**2) - 40*x[1]
	H[0,1]= -40*x[0]
	H[1,0] = H[0,1]
	H[1,1] = 20
	return H

x = np.array([1,1])

F_r = evalFR(x)
G = gradFR(x)
H = hessFR(x)

"""
print(F_r)
print(G)
print(H)
"""

def step_research(x,d,rho,epsilon,grad,hess,itmax):
	j = 1 
	rho_1 = 10*rho
	while (np.linalg.norm(rho_1 - rho)>= epsilon and j<itmax):
		rho = rho_1
		rho_p = np.transpose(d)@grad(x+rho*d)
		rho_pp = np.transpose(d)@hess(x+rho*d)@d
		rho_1 = rho - (rho_p/rho_pp)
		j +=1

	return rho_1

def fixed_step_gradient(x0, rho, epsilon,itmax, grad):
	i = 1
	x = x0
	xit = [x0]
	while(np.linalg.norm(grad(x))>epsilon and i < itmax):
		d = -grad(x)
		x = x + rho*d
		i += 1 
		xit.append(x)

	return (x,xit,i)

def optimal_step_gradient(x0, epsilon, grad, hess, itmax):
	i = 1
	x = x0
	rho = 1e-5
	xit = [x]
	while (np.linalg.norm(grad(x))>epsilon and i < itmax):
		d = -grad(x)
		rho = step_research(x,d,rho, epsilon, grad, hess,itmax)
		x = x+rho*d
		i += 1
		xit.append(x)

	return (x,xit,i)

def optimality_conditions(xNum, epsilon,grad,hess):

	Gr = grad(xNum)
	print(Gr)
	He = hess(xNum)
	lbd, v = np.linalg.eig(He)
	print(lbd)
	mini = list()
	for i in range(len(lbd)):
		if (np.linalg.norm(Gr)<epsilon and lbd[i] >= 0):
			mini.append(True)
		else :
			mini.append(False)

	return Gr,lbd,mini

itmax = 1e6


"""
n = 9
x0_1 = np.random.randint(0,200,n)
rho = 10e-5
start_t1 = time.time()
xs, xits,ni = fixed_step_gradient(x0_1, rho, 10e-6, itmax, gradFQ)
end_t1 = time.time()
time1 = end_t1 - start_t1
print(xs, ni, time1)
"""


"""
n = 81
x0_1 = np.random.randint(0,200,n)
#x0 = np.array([2,2])
print("-------- Quadratic function-----------")
start_t1 = time.time()
xs, xits, ni = optimal_step_gradient(x0_1, 10e-6, gradFQ, HessFQ, itmax)
end_t1 = time.time()
time1 = end_t1 - start_t1
print("solution : ", xs)
print("itération number : ", ni)
print("CPU time : ", time1, "s")

N = [9,25,81,144]

N=[2]
NI = list()
TIME = list()
NI2 = list()
TIME2 = list()
for n_ in N:
	x0_1 = np.random.randint(0,200,n_)
	
	#x0 = np.array([2,2])
	print("-------- Quadratic function-----------")

	start_t1 = time.time()
	xs, xits, ni = optimal_step_gradient(x0_1, 10e-6, gradFQ, HessFQ, itmax)
	end_t1 = time.time()
	time1 = end_t1 - start_t1

	start_t2 = time.time()
	xs2, xits2, ni2 = fixed_step_gradient(x0_1, 1, 10e-6, itmax, gradFQ)
	end_t2 = time.time()
	time2 = end_t2 - start_t2

	NI.append(ni)
	TIME.append(time1)
	
	NI2.append(ni2)
	TIME2.append(time2)

xits2_ = np.array(xits2)

xits_ = np.array(xits)

for i in range(len(NI)):

	print("Taille n : ", N[0])
	print ("----------Optimal step gradient---------")
	print("x0 :", x0_1)
	print("solution : ", xits_[-1,:])
	print("itération number : ", NI[-1])
	print("CPU time : ", TIME[-1], "s")

	print ("----------Fixed step gradient---------")
	print("x0 : ", x0_1)
	print("solution : ", xits2_[-1,:])
	print("itération number : ", NI2[-1])
	print("CPU time : ", TIME2[-1], "s")


plt.figure("iteration ")
plt.loglog(N,NI,label= "optimal step gradient",color ="green")
plt.loglog(N, NI2, label= "fixed step gradient", color = "crimson")
plt.legend()
plt.title("iteration in function of N")


plt.figure("time")
plt.loglog(N,TIME,label= "optimal step gradient",color ="green")
plt.loglog(N,TIME2, label="fixed step gradient", color = "crimson")
plt.legend()
plt.title("time in function of N")
plt.show()


x0_2 = np.array([2,2])

print("-------- Convex function-----------")

start_t2 = time.time()
xs2, xits2, ni2 = optimal_step_gradient(x0_2, 10e-6, gradFC, HessFC, itmax)
end_t2 = time.time()
time2 = end_t2 - start_t2


start_t22 = time.time()
xs22, xits22, ni22 = fixed_step_gradient(x0_2, 1e-4, 10e-6, itmax,gradFC)
end_t22 = time.time()
time22 = end_t22 - start_t22

xits2_ = np.array(xits2)
xits22_ = np.array(xits22)

plt.figure("Level map 2D de g")
plt.plot(xits2_[:,0], xits2_[:,1], color = 'blue', label='optimal step gradient')
plt.plot(xits22_[:,0], xits22_[:,1], color = 'green', label='fixed step gradient')
lm2D_2 = plt.contour(x_axis_2,y_axis_2, gtest, 100, cmap ="hot")
plt.title("Lignes de niveaux de g")
plt.colorbar(lm2D_2)
plt.legend()
plt.show()


GS1 = list()
GS2 = list()
I = list()
I2 = list()

for i in range(len(xits2_)):
	x1 = xits2_[i,0]
	x2 =xits2_[i,1]
	gs1 = g(x1,x2)
	GS1.append(gs1)
	print(GS1)
	I.append(i)
	

for i2 in range(len(xits22_)):
	x1 = xits22_[i2,0]
	x2 =xits22_[i2,1]
	gs2 = g(x1,x2)
	GS2.append(gs2)
	I2.append(i2)


print("-------- optimal-----------")
print("solution : ", xs2)
print("itération number : ", ni2)
print("CPU time : ", time2, "s")

print("g value : ", GS1[-1])

print("--------------------------------------")


Gr, lbd, mini = optimality_conditions(xs2, 1e-4, gradFC, HessFC)
if False not in mini :
	print(xs2, "est un minimum local")

print("-------- fixed-----------")
print("solution : ", xs22)
print("itération number : ", ni22)
print("CPU time : ", time22, "s")
print("g value : ", GS2[-1])
print("--------------------------------------")

plt.figure("solution evolution")
plt.semilogx(I, GS1, color = 'mediumpurple', label='optimal_step_gradient')
plt.semilogx(I2, GS2, color = 'turquoise', label='fixed_step_gradient')
plt.title("solution")
plt.legend()
plt.show()


print("-------- Rosembrok function-----------")

x0_2 = np.array([1.2,1])


start_t3 = time.time()
xs3, xits3, ni3 = optimal_step_gradient(x0_2, 10e-6, gradFR, hessFR, itmax)
end_t3 = time.time()
time3 = end_t3 - start_t3

start_t32 = time.time()
xs32, xits32, ni32 = fixed_step_gradient(x0_2,1e-3, 10e-6, itmax, gradFR)
end_t32 = time.time()
time32 = end_t32 - start_t32


xits3_ = np.array(xits3)
xits32_ = np.array(xits32)

plt.figure("Level map 2D de Rosembrok")
plt.title("Lignes de niveaux de Rosembrok")
plt.plot(xits3_[:,0], xits3_[:,1], color = 'orange', label='optimal step gradient')
plt.plot(xits32_[:,0], xits32_[:,1], color = 'red', label='fixed step gradient')
lm2D_r = plt.contour(x_axis_2,y_axis_2, r, 100, cmap="plasma")
plt.colorbar(lm2D_r)
plt.legend()
plt.show()

GS1 = list()
GS2 = list()
I = list()
I2 = list()

for i in range(len(xits3_)):
	x1 = xits3_[i,0]
	x2 =xits3_[i,1]
	gs1 = g(x1,x2)
	GS1.append(gs1)
	print(GS1)
	I.append(i)
	

for i2 in range(len(xits32_)):
	x1 = xits32_[i2,0]
	x2 =xits32_[i2,1]
	gs2 = g(x1,x2)
	GS2.append(gs2)
	I2.append(i2)




print('---------Optimal------------')
print("solution : ", xs3)
print("itération number : ", ni3)
print("CPU time : ", time3, "s")
print("Rosembrock value : ", GS1[-1])
print('---------fixed------------')
print("solution : ", xs32)
print("itération number : ", ni32)
print("CPU time : ", time32, "s")
print("Rosembrock value : ", GS2[-1])
print("--------------------------------------")


plt.figure("solution evolution")
plt.semilogx(I, GS1, color = 'midnightblue', label='optimal_step_gradient')
plt.semilogx(I2, GS2, color = 'olive', label='fixed_step_gradient')
plt.title("solution")
plt.legend()
plt.show()

Gr, lbd, mini = optimality_conditions(xs3, 1e-4, gradFR, hessFR)
if False not in mini :
	print(xs3, "est un minimum local")





#point selle

def fbonus1(x1,x2):
	return(x1**2 - x2**2)


def Gradfbonus(x):
	G = np.zeros(2)
	G[0] = 2*x[0]
	G[1] = - 2*x[1]

	return G

def Hessfbonus(x):
	H = np.zeros((2,2))
	H[0,0] = 2
	H[0,1] = 0
	H[1,0] = H[0,1]
	H[1,1] = -H[0,0]

	return H

xstar = np.array([0,0])

H = Hessfbonus(xstar)
G = Gradfbonus(xstar)
A = H
b = A@xstar - G 

d1 = np.array([1,0])
d2 = np.array([0,1])

def partial_function (d, cstar, A, b):
	t = np.arange(-3,3.5, 0.05)
	F_cstar = list()
	for i in t: 
		f_cstar = (1/2)* (np.transpose(d)@A@d*(i**2))-np.transpose((b - A@cstar))@d*i + (1/2)*(np.transpose(cstar)@A@cstar)-np.transpose(b)@cstar
		F_cstar.append(f_cstar)

	return(F_cstar ,t)

F_xstar1, t1 = partial_function(d1,xstar,A,b)
F_xstar2, t2 = partial_function(d2,xstar,A,b)

plt.figure("Partials functions")
plt.subplot(2,2,1)
plt.plot(t1,F_xstar1, color ='deepskyblue')
plt.title("Courbe de la fonction partielle $F_{x^*,d}$ pour $d=e_1$")
plt.subplot(2,2,2)
plt.plot(t2,F_xstar2, color ='springgreen')
plt.title("Courbe de la fonction partielle $F_{x^*,d}$ pour $d=e_2$")
plt.show()


#Pc ni extrem ni ps

def fbonus2(x1,x2):
	return(x1**2 - x2**2)


def Gradfbonus2(x):
	G = np.zeros(2)
	G[0] = 3*x[0]
	G[1] = - 2*x[1]

	return G

def Hessfbonus2(x):
	H = np.zeros((2,2))
	H[0,0] = 3
	H[0,1] = 0
	H[1,0] = H[0,1]
	H[1,1] = -2

	return H

xstar = np.array([0,0])

H = Hessfbonus2(xstar)
G = Gradfbonus2(xstar)
A = H
b = A@xstar - G 

d1 = np.array([1,0])

def partial_function (d, cstar, A, b):
	t = np.arange(-3,3.5, 0.05)
	F_cstar = list()
	for i in t: 
		f_cstar = (1/2)* (np.transpose(d)@A@d*(i**2))-np.transpose((b - A@cstar))@d*i + (1/2)*(np.transpose(cstar)@A@cstar)-np.transpose(b)@cstar
		F_cstar.append(f_cstar)

	return(F_cstar ,t)

F_xstar1, t1 = partial_function(d1,xstar,A,b)
plt.figure("Partial function")
plt.plot(t1,F_xstar1, color ='deepskyblue')
plt.title("Courbe de la fonction partielle $F_{x^*,d}$ pour $d=e_1$")
plt.show()


"""







