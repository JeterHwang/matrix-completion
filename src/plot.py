import matplotlib.pyplot as plt
import torch
import numpy as np
from src.utils import emb2polar, emb2rect, polar2emb, rect2emb

def plot_loss(fig_path, losses):
    fig, ax = plt.subplots(3, 2, figsize=(20, 30), tight_layout=True)
    loss    = [loss[0] for loss in losses]
    diff    = [loss[1] for loss in losses]
    grad_X  = [loss[2] for loss in losses]
    hess_X  = [loss[3] for loss in losses]
    f_X     = [loss[4] for loss in losses]
    f_Z     = [loss[5] for loss in losses]
    trans   = [loss[6] for loss in losses]
    
    ax[0][0].plot(loss)
    ax[0][1].plot(diff)
    ax[1][0].plot(grad_X)
    ax[1][1].plot(trans)
    ax[2][0].plot(hess_X)
    ax[2][1].plot(f_X, label=r'f(X)')
    ax[2][1].plot(f_Z, label=r'f(Z)')

    ax[0][0].set_xlabel("# of GD Updates", fontsize=30), 
    ax[0][0].set_ylabel(r"$g(X,Z,E)$", fontsize=30)
    ax[0][0].set_title(f"Total Loss vs. # of GD Updates", fontsize=25, fontname='Comic Sans MS')
    ax[0][0].xaxis.set_tick_params(labelsize=20)
    ax[0][0].yaxis.set_tick_params(labelsize=20)
    ax[0][0].grid()

    ax[0][1].set_xlabel("# of GD Updates", fontsize=30), 
    ax[0][1].set_ylabel(r"$f(Z)-f(X)$", fontsize=30)
    ax[0][1].set_title(r"$f(Z)-f(X)$ vs. # of GD Updates", fontsize=25, fontname='Comic Sans MS')
    ax[0][1].xaxis.set_tick_params(labelsize=20)
    ax[0][1].yaxis.set_tick_params(labelsize=20)
    ax[0][1].grid()

    ax[1][0].set_xlabel("# of GD Updates", fontsize=30), 
    ax[1][0].set_ylabel(r"$\|\nabla_Xf(X)\|_F$", fontsize=30)
    ax[1][0].set_title(r"$\|\nabla_Xf(X)\|_F$ vs. # of GD Updates", fontsize=25, fontname='Comic Sans MS')
    ax[1][0].xaxis.set_tick_params(labelsize=20)
    ax[1][0].yaxis.set_tick_params(labelsize=20)
    ax[1][0].set_yscale('log')
    ax[1][0].grid()

    ax[1][1].set_xlabel("# of GD Updates", fontsize=30), 
    ax[1][1].set_ylabel(r"$-\|XX^T-ZZ^T\|_F$", fontsize=30)
    ax[1][1].set_title(r"$-\|XX^T-ZZ^T\|_F$ vs. # of GD Updates", fontsize=25, fontname='Comic Sans MS')
    ax[1][1].xaxis.set_tick_params(labelsize=20)
    ax[1][1].yaxis.set_tick_params(labelsize=20)
    ax[1][1].grid()

    ax[2][0].set_xlabel("# of GD Updates", fontsize=30), 
    ax[2][0].set_ylabel(r"$-\lambda_{min}(\nabla^2_Xf(X))$", fontsize=30)
    ax[2][0].set_title(r"$-\lambda_{min}(\nabla^2_Xf(X))$ vs. # of GD Updates", fontsize=25, fontname='Comic Sans MS')
    ax[2][0].xaxis.set_tick_params(labelsize=20)
    ax[2][0].yaxis.set_tick_params(labelsize=20)
    # ax[2][0].set_yscale('log')
    ax[2][0].grid()

    ax[2][1].set_xlabel("# of GD Updates", fontsize=30), 
    ax[2][1].set_ylabel("Function Value", fontsize=30)
    ax[2][1].set_title("Function Value vs. # of GD Updates", fontsize=25, fontname='Comic Sans MS')
    ax[2][1].xaxis.set_tick_params(labelsize=20)
    ax[2][1].yaxis.set_tick_params(labelsize=20)
    # ax[2][1].set_yscale('log')
    ax[2][1].legend(loc='best')
    ax[2][1].grid()

    fig.savefig(fig_path, dpi=300)

def plot_contour(
    fig_path,
    X_emb       : torch.Tensor, # in embedding form
    Z_emb       : torch.Tensor, # in embedding form
    criterion,
    constraints : tuple,
    res         : int           = 100,
):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20), tight_layout=True)
    levels = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    mag = np.linspace(constraints[0], constraints[1], res)
    ang = np.linspace(constraints[2], constraints[3], res)
    x, y = np.meshgrid(ang * 180 / np.pi, mag)
    
    # plot X contour
    X_polar = emb2polar(X_emb, *constraints)
    V1 = X_polar[0].item()
    #print(mag)
    # print(ang)
    F = np.zeros((res, res))
    for i, V in enumerate(mag):
        for j, Th in enumerate(ang):
            polar = torch.tensor([V1, V, Th], dtype=torch.float64).reshape((3, 1))
            F[i][j] = criterion(polar2emb(polar, *constraints)).item()
    ax[0][0].plot(X_polar[2] / torch.pi * 180, X_polar[1], 'g*', markersize=10, label='X')
    CS = ax[0][0].contour(x, y, F, levels)
    ax[0][0].clabel(CS, fontsize=10, fmt="%.2f")
    ax[0][0].set_title(r"$V_1=$" + f"{V1:.2f} (p.u.)", fontsize=25, fontname='Comic Sans MS')
    ax[0][0].set_xlabel(r"$\theta_2$", fontsize=20), 
    ax[0][0].set_ylabel(r"$V_2$ (p.u.)", fontsize=20)
    ax[0][0].xaxis.set_tick_params(labelsize=15)
    ax[0][0].yaxis.set_tick_params(labelsize=15)
    ax[0][0].legend(loc='best')
    
    # plot Z contour
    Z_polar = emb2polar(Z_emb, *constraints)
    V1 = Z_polar[0].item()
    for i, V in enumerate(mag):
        for j, Th in enumerate(ang):
            polar = torch.tensor([V1, V, Th], dtype=torch.float64).reshape((3, 1))
            F[i][j] = criterion(polar2emb(polar, *constraints)).item()
    ax[0][1].plot(Z_polar[2] / torch.pi * 180, Z_polar[1], 'r*', markersize=10, label='Z')
    CS = ax[0][1].contour(x, y, F, levels)
    ax[0][1].clabel(CS, fontsize=10, fmt="%.2f")
    ax[0][1].set_title(r"$V_1=$" + f"{V1:.2f} (p.u.)", fontsize=25, fontname='Comic Sans MS')
    ax[0][1].set_xlabel(r"$\theta_2$", fontsize=20), 
    ax[0][1].set_ylabel(r"$V_2$ (p.u.)", fontsize=20)
    ax[0][1].xaxis.set_tick_params(labelsize=15)
    ax[0][1].yaxis.set_tick_params(labelsize=15)
    ax[0][1].legend(loc='best')
    
    # plot 2D surface
    X_rect, Z_rect = emb2rect(X_emb, *constraints), emb2rect(Z_emb, *constraints)
    U = np.linspace(-1.5, 1.5, res)
    V = np.linspace(-1.5, 1.5, res)
    levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    x, y = np.meshgrid(V, U)
    for i, u in enumerate(U):
        for j, v in enumerate(V):
            rect = u * X_rect + v * Z_rect
            F[i][j] = criterion(rect2emb(rect, *constraints)).item()
    ax[1][0].plot(1, 0, 'r*', markersize=10, label='Z')
    ax[1][0].plot(0, 1, 'g*', markersize=10, label='X')
    CS = ax[1][0].contour(x, y, F, levels)
    ax[1][0].clabel(CS, fontsize=10, fmt="%.2f")
    ax[1][0].set_title(r"$F(u \times Z + v \times X)$", fontsize=25, fontname='Comic Sans MS')
    ax[1][0].set_xlabel(r"$u$", fontsize=20), 
    ax[1][0].set_ylabel(r"$v$", fontsize=20)
    ax[1][0].xaxis.set_tick_params(labelsize=15)
    ax[1][0].yaxis.set_tick_params(labelsize=15)
    ax[1][0].legend(loc='best')

    fig.savefig(fig_path, dpi=300)
