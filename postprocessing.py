import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import os
import imageio
# Read dimension data from configuration file
config_f = open(f"out/config_file.dat")
datalist = []
for i in config_f.readlines():
    l = i.split()
    datalist.append(float(l[1]))
Nx = int(datalist[0])
Ny = int(datalist[1])
Lx = datalist[2]
Ly = datalist[3]
dimx = int(datalist[4])
dimy = int(datalist[5])
# hence the mesh spacing
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
lenx = int(Nx//dimx)
leny = int(Ny//dimy)
X, Y = np.mgrid[0:Nx:1, 0:Ny:1]
X = dx*X
Y = dy*Y
try:
    os.mkdir("img")
except OSError as error:
    print(error)
try:
    os.mkdir("result_data")
except OSError as error:
    print(error)
try:
    os.mkdir("gifs")
except OSError as error:
    print(error)
allfile = os.listdir("out")
fileinfo = pd.DataFrame(columns=['id', 'feild', 'time', 'name'])
for i in allfile:
    temp = i.split(".")[0].split("_")
    if temp[0] != 'id':
        continue
    fileinfo.loc[len(fileinfo)] = [int(temp[1]), temp[2], int(temp[3]), i]
for it in range(max(fileinfo['time'])+1):
    cur_df = fileinfo[fileinfo['time'] == it]
    if len(cur_df) != dimx*dimy*3:
        raise Exception(f"missing file for time {it}")
    for feildname in ['P', 'u', 'v']:
        feild_df = cur_df[cur_df["feild"] == feildname]
        toplot = np.zeros([Nx, Ny])
        if len(feild_df) != dimx*dimy:
            raise Exception(f"missing file for feild {feildname}")
        for j, process in feild_df.iterrows():
            cur_file = open("out/"+process["name"], "r")
            data = []
            for i in cur_file.readlines():
                temp = [float(j) for j in i.split()]
                data.append(temp)
            data = np.array(data)
            idx_x = process["id"] % dimx
            idx_y = process["id"]//dimx
            start_x = idx_x*lenx+1
            start_y = idx_y*leny+1
            curlen_x = len(data)-2
            curlen_y = len(data[0])-2
            toplot[start_x:start_x+curlen_x,
                   start_y:start_y+curlen_y] = data[1:-1, 1:-1]
            if (idx_x == 0):
                toplot[0, start_y-1:start_y+curlen_y+1] = data[0, :]
            if (idx_x == dimx-1):
                toplot[-1, start_y-1:start_y+curlen_y+1] = data[-1, :]
            if (idx_y == dimy-1):
                toplot[start_x-1:start_x+curlen_x+1, -1] = data[:, -1]
            if (idx_y == 0):
                toplot[start_x-1:start_x+curlen_x+1, 0] = data[:, 0]
            cur_file.close()
        combined = open(f"result_data/{feildname}_{it}.dat", 'w')
        np.savetxt(combined, toplot, fmt='%.8f')
        combined.close()

        fig = plt.figure(figsize=(21, 7))
        ax1 = fig.add_subplot(121)
        cont = ax1.contourf(X, Y, toplot, cmap=cm.coolwarm)
        fig.colorbar(cont)
        # # don't plot at every gird point - every 5th
        # ax1.quiver(X[::20,::5],Y[::20,::5],uc[::20,::5],vc[::20,::5])
        ax1.set_xlim(0, 0.1)
        ax1.set_ylim(0, 0.05)
        ax1.set_xlabel('$x$', fontsize=16)
        ax1.set_ylabel('$y$', fontsize=16)
        ax1.set_title(
            f'Pressure driven problem - {feildname} - timestep{it}', fontsize=16)
        plt.savefig(f'./img/img_{feildname}_{it}.png',
                    transparent=False, facecolor='white')
        plt.close()
