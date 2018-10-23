n = 20

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

aux = ponto0
for i in range(1,n):
    aux = ponto0[:]
    for j in range(n-i+1):
        aux = ponto0[:]
        for k in range(j, j+i):
            aux[k] = 1
        print(str(aux)+"\n")
