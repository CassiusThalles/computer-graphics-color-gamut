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

#Gerando a lista com todos os valores Vm a serem utilizados:
n = 401

pontos = []
pontos1 = []
pontos2 = []
ponto0 = []
ponto1 = []
pontosaux = []
pontosaux2 = []

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
        g+=1
        e+=1

        k = 0
        aux1 = 0
        aux2 = 0
        aux3 = 0
        aux4 = 0
        aux5 = 0

        # calculando efetivamente as coordenadas XYZ
        for j in range(401):
            aux1 = aux1 + (param[j + 80] * aux[j] * (xbar[j + 80] + ybar[j + 80] + zbar[j + 80]))
            aux2 = aux2 + (param[j + 80] * aux[j] * xbar[j + 80])
            aux3 = aux3 + (param[j + 80] * aux[j] * ybar[j + 80])
            aux4 = aux4 + (param[j + 80] * aux[j] * zbar[j + 80])
            aux5 = aux5 + (param[j + 80] * ybar[j + 80])

        xp = aux2 / aux1
        yp = aux3 / aux1
        zp = 1 - (xp + yp)

        k = 100 / aux5
        X = k * aux2
        Y = k * aux3
        Z = k * aux4

        format(X, '6f')
        format(Y, '6f')
        format(Z, '6f')
        format(xp, '6f')
        format(yp, '6f')
        format(zp, '6f')

        # calculando sRGB

        R = 3.2404542 * xp - 1.5371385 * yp - 0.4985314 * zp
        G = -0.9692660 * xp + 1.8760108 * yp + 0.0415560 * zp
        B = 0.0556434 * xp - 0.2040259 * yp + 1.0572252 * zp

        if (R <= 0.04045):
            R = 12.92 * R
        elif (R > 0.04045):
            R = 1.055 * (R ** (1 / 2.4) - 0.055)

        if (G <= 0.04045):
            G = 12.92 * G
        elif (G > 0.04045):
            G = 1.055 * (G ** (1 / 2.4) - 0.055)

        if (B <= 0.04045):
            B = 12.92 * B
        elif (B > 0.04045):
            B = 1.055 * (B ** (1 / 2.4) - 0.055)

        if (R < 0):
            R = 0.0
        elif (R > 1):
            R = 1.0

        if (G < 0):
            G = 0.0
        elif (G > 1):
            G = 1.0

        if (B < 0):
            B = 0.0
        elif (B > 1):
            B = 1.0

        pontoSRGB = [R, G, B]

        ax.scatter(R, G, B, c=pontoSRGB, marker=',', s=2)
        print("ok " + str(d) + "\n")
        d += 1

    d+=1

d=0
e=0
f=0
g=0
h=0

while(d<=i):
    e = 0 + d
    aux = ponto1[:]
    while(e<=i):
        g = 0+e
        aux[d] = 0
        aux[g] = 0
        pontosaux2.append(aux[:])
        g+=1
        e+=1
    d+=1

d = 0
e = 80
aux = []
while(d<80):

d = 0



ax.scatter(0, 0, 0, c='black', marker=',', s=2)
ax.scatter(1, 1, 1, c='white', marker=',', s=2)

plt.show()