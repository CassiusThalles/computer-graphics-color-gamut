import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color

#A função tratamentoDados tratará apenas arquivos em excel
#informação 1: ao passar o argumento na função deve ser passado com as aspas de string

data = pd.read_excel("all_1nm_data.xls", header=0, skiprows=3)
data = data.values.T.tolist()

#tratando os dados de data
a = len(data)
b = len(data[1])

for i in range(a):
    data[i] = np.array(data[i])

#trocando os valores "nam" por 0
for i in range(a):
    vm_nam_i = np.isnan(data[i])
    data[i][vm_nam_i] = 0

#função para transformar rgb em hexadecimal
def rgb2hex(r, g, b):
    hex = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    return hex

#Gerando a lista com todos os valores Vm a serem utilizados:
n = 401

pontos = []
ponto0 = []
ponto1 = []
pontosaux = []

d = 0
e = 0
f = 0
g = 0
h = 0
i = n-1

while(d<n):
    ponto0.append(0)
    d+=1
d=0
while(d<n):
    ponto1.append(1)
    d+=1
d=0

while(d<=i):
    e = 0 + d
    aux = ponto0[:]
    while(e<=i):
        g = 0+e
        aux[d] = 1
        aux[g] = 1
        pontosaux.append(aux[:])
        g+=1
        e+=1
    d+=1
#pontosaux.append(ponto0)
#pontosaux.append(ponto1)

d = 0
e = 80
aux = []
while(d<80):
    aux.append(0)
    d+=1

for vetor in pontosaux:
    aux2 = []
    aux2.extend(aux[:])
    aux2.extend(vetor[:])
    pontos.append(aux2[:])

d = len(pontos[0])
e = 531
f = e-d

for vetor in pontos:
    for escalar in range(f):
        vetor.append(0)

#Calculando as coordenadas XYZ
#valores auxiliares

ciea = data[1]
cied65 = data[2]
xbar = data[5]
ybar = data[6]
zbar = data[7]
xbar1 = data[8]
ybar1 = data[9]
zbar1 = data[10]

param = cied65

#iniciando o gráfico onde os pontos serão plotados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

d = 0

for vm in pontos:
    k = 0
    aux1 = 0
    aux2 = 0
    aux3 = 0
    aux4 = 0
    aux5 = 0

    # calculando efetivamente as coordenadas XYZ
    for i in range(531):
        aux1 = aux1 + (param[i] * vm[i] * (xbar[i] + ybar[i] + zbar[i]))
        aux2 = aux2 + (param[i] * vm[i] * xbar[i])
        aux3 = aux3 + (param[i] * vm[i] * ybar[i])
        aux4 = aux4 + (param[i] * vm[i] * zbar[i])
        aux5 = aux5 + (param[i] * ybar[i])


    xp = aux2 / aux1
    yp = aux3 / aux1
    zp = 1 - (xp + yp)

    k = 100 / aux5
    X = k * aux2
    Y = k * aux3
    Z = k * aux4

    #calculando sRGB

    R = 3.2404542 * xp - 1.5371385 * yp - 0.4985314 * zp
    G = -0.9692660 * xp + 1.8760108 * yp + 0.0415560 * zp
    B = 0.0556434 * xp - 0.2040259 * yp + 1.0572252 * zp

    R1 = 0.0
    G1 = 0.0
    B1 = 0.0

    auxList = [R, G, B]

    pontoSRGB = []

    for v in auxList:
        if (v <= 0.04045):
            V = 12.92 * v
            pontoSRGB.append(V)
        elif (v > 0.04045):
            V = 1.055 * (v ** (1 / 2.4) - 0.055)
            pontoSRGB.append(V)

    Rf = 255.0 * auxList[0]
    Gf = 255.0 * auxList[1]
    Bf = 255.0 * auxList[2]

    ponto = [Rf, Gf, Bf]

    if (pontoSRGB[0] < 0):
        pontoSRGB[0] = 0.0
    elif (pontoSRGB[0] > 1):
        pontoSRGB[0] = 1.0

    if (pontoSRGB[1] < 0):
        pontoSRGB[1] = 0.0
    elif (pontoSRGB[1] > 1):
        pontoSRGB[1] = 1.0

    if (pontoSRGB[2] < 0):
        pontoSRGB[2] = 0.0
    elif (pontoSRGB[2] > 1):
        pontoSRGB[2] = 1.0

    ax.scatter(X, Y, Z, c=pontoSRGB, marker=',', s=2)
    print("Cálculos completos: "+str(d)+" de 80600\n")
    d+=1

ax.scatter(0, 0, 0, c='black', marker=',', s=2)
ax.scatter(1, 1, 1, c='white', marker=',', s=2)

plt.show()