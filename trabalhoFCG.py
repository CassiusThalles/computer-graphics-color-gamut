import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

data = pd.read_excel('all_1nm_data.xls', header=0, skiprows=3)

#tratando a coluna nm
nm = data['nm'].tolist()
nm = np.array(data['nm'])

#tratando a coluna CIE A
ciea = data['CIE A'].tolist()
ciea = np.array(data['CIE A'])

#tratando a coluna CIE D65
cied65 = data['CIE D65'].tolist()
cied65 = np.array(data['CIE D65'])

#tratando a coluna VM(l)
vm = data['VM(l)'].tolist()
vm = np.array(data['VM(l)'])

#tratando a coluna V'(l)
vl = data["V'(l)"].tolist()
vl = np.array(data["V'(l)"])

#tratando a coluna x bar
x = data['x bar'].tolist()
x = np.array(data['x bar'])

#tratando a coluna y bar
y = data['y bar'].tolist()
y = np.array(data['y bar'])

#tratando a coluna z bar
z = data['z bar'].tolist()
z = np.array(data['z bar'])

#tratando a coluna x bar.1
x1 = data['x bar.1'].tolist()
x1 = np.array(data['x bar.1'])

#tratando a coluna y bar.1
y1 = data['y bar.1'].tolist()
y1 = np.array(data['y bar.1'])

#tratando a coluna z bar.1
z1 = data['z bar.1'].tolist()
z1 = np.array(data['z bar.1'])

#trocando os campos nan por 0's
vmNam = np.isnan(vm)
vm[vmNam] = 0

vlNam = np.isnan(vl)
vl[vlNam] = 0

xNam = np.isnan(x)
x[xNam] = 0

yNam = np.isnan(y)
y[yNam] = 0

zNam = np.isnan(z)
z[zNam] = 0

x1Nam = np.isnan(x1)
x1[x1Nam] = 0

y1Nam = np.isnan(y1)
y1[y1Nam] = 0

z1Nam = np.isnan(z1)
z1[z1Nam] = 0

#fazendo os cálculos para as coordenadas com base no iluminante CIE A
#Calculando as coordenadas XYZ utilizando Vm(l)

print("Para calcular as coordenadas XYZ da cor é necessário que seja informado alguns parâmetros.")

param1 = input("Qual o iluminante a ser usado? CIE A (a) ou CIE D65(d65)?")
param2 = input("Qual o ângulo de visualização será utilizado? 2 ou 10 graus?")
print("\n\n")

k = 0
X = 0
Y = 0
Z = 0

#criando as variáveis auxiliares
aux1 = 0
aux2 = 0
aux3 = 0
aux4 = 0
aux5 = 0

if(param1 == "a"):
    param = ciea
elif(param1 == "d65"):
    param = cied65

if(param2 == "2"):
    for i in range(531):
        aux1 = aux1 + (param[i]*vm[i]*(x[i]+y[i]+z[i]))
        aux2 = aux2 + (param[i]*vm[i]*x[i])
        aux3 = aux3 + (param[i]*vm[i]*y[i])
        aux4 = aux4 + (param[i]*vm[i]*z[i])
        aux5 = aux5 + (param[i]*y[i])
elif(param2 == "10"):
    for i in range(531):
        aux1 = aux1 + (param[i]*vm[i]*(x1[i]+y1[i]+z1[i]))
        aux2 = aux2 + (param[i]*vm[i]*x1[i])
        aux3 = aux3 + (param[i]*vm[i]*y1[i])
        aux4 = aux4 + (param[i]*vm[i]*z1[i])
        aux5 = aux5 + (param[i]*y1[i])

xp = aux2/aux1
yp = aux3/aux1
zp = 1 - (xp + yp)

k = 100/aux5
X = k*aux2
Y = k*aux3
Z = k*aux4

pontoXYZ = [xp, yp, zp]
print("As coordenadas XYZ do ponto são: " + str(pontoXYZ)+"\n")
print("X= " + str(X) + " Y= " + str(Y) + " Z= " + str(Z))

print("Calculando agora as coordendas CIE L*a*b*")

e = 0.008856
k = 903.3
Xr = 0
Yr = 0
Zr = 0

if(param1 == "a"):
    if(param2 == "2"):
        Xr = 109.847
        Yr = 100.00
        Zr = 35.582
    elif(param2 == "10"):
        Xr = 111.142
        Yr = 100.00
        Zr = 35.200
elif(param1 == "d65"):
    if(param2 == "2"):
        Xr = 95.043
        Yr = 100.00
        Zr = 108.890
    elif(param2 == "10"):
        Xr = 94.810
        Yr = 100.00
        Zr = 107.305

xr = X/Xr
yr = Y/Yr
zr = Z/Zr

fx = 0
fy = 0
fz = 0

if(xr > e):
    fx = xr**(1/3)
elif(xr <= e):
    fx = (k*xr +16)/116

if(yr > e):
    fy = yr**(1/3)
elif(yr<=e):
    fy = (k*yr + 16)/116

if(zr > e):
    fz = zr**(1/3)
elif(zr <= e):
    fz = (k*zr + 16)/116

L = 116*fy - 16
a = 500*(fx - fy)
b = 200*(fy- fz)

pontoLab = [L, a, b]

print("As coordenadas La*b* da cor em questão são " + str(pontoLab) + "\n")

print("Calculando as coordenadas sRGB")

if(param1 == "d65"):
    R = 3.2404542*xp -1.5371385*yp -0.4985314*zp
    G = -0.9692660*xp +1.8760108*yp +0.0415560*zp
    B = 0.0556434*xp -0.2040259*yp +1.0572252*zp
elif(param1 == "a"):
    R = 3.2404542*xp -1.5371385*yp -0.4985314*zp
    G = -0.9692660*xp +1.8760108*yp +0.0415560*zp
    B = 0.0556434*xp -0.2040259*yp +1.0572252*zp

R1 = 0.0
G1 = 0.0
B1 = 0.0

auxList= [R, G, B]

pontoSRGB = []

for v in auxList:
    if(v <=0.04045):
        V = 12.92*v
        pontoSRGB.append(V)
    elif(v > 0.04045):
        V = 1.055*(v**(1/2.4) - 0.055)
        pontoSRGB.append(V)



Rf = 255.0*auxList[0]
Gf = 255.0*auxList[1]
Bf = 255.0*auxList[2]

ponto = [Rf, Gf, Bf]

for i in range(3):
    if(ponto[i] < 0):
        ponto[i] = 0.0000000000000
    elif(ponto[i] > 255):
        ponto[i] = 255.0000000000000

print("As corrdenadas da cor no sistema sRGB são: " + str(ponto))

def rgb2hex(r,g,b):
    hex = "#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))
    return hex

print(rgb2hex(Rf,Gf,Bf))