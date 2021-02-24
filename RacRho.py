import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io
import time


plt.rc("text", usetex=False)
plt.rc("font", family="sans-serif", size=12)


def f(R,Ri,rho,rhoi):
    return (1/(1+rho**4))*Ri - R

def g(R,Ri,rho,rhoi):
    return (1/(1+R**4))*rhoi - rho

def rdPDE(t, y, D, Di, dx):
    """
    The ODEs are derived using the method of lines.
    https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol
    """
    # The vectors R, Ri, rho, rhoi are interleaved in y for computational speed
    # every fourth entry is...
    R = y[0::4]
    Ri = y[1::4]
    rho = y[2::4]
    rhoi = y[3::4]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    dRdt = dydt[0::4]
    dRidt = dydt[1::4]
    drhodt = dydt[2::4]
    drhoidt = dydt[3::4]

    dRdt[0]    = f(R[0],    Ri[0],    rho[0],    rhoi[0]) +  D * (-2.0*R[0] + 2.0*R[1]) / dx**2
    dRdt[1:-1] = f(R[1:-1], Ri[1:-1], rho[1:-1], rhoi[1:-1]) +  D * np.diff(R,2) / dx**2
    dRdt[-1]   = f(R[-1],   Ri[-1],   rho[-1],   rhoi[-1]) +  D * (-2.0*R[-1] + 2.0*R[-2]) / dx**2

    dRidt[0]    = -f(R[0],    Ri[0],    rho[0],    rhoi[0]) +  Di * (-2.0*Ri[0] + 2.0*Ri[1]) / dx**2
    dRidt[1:-1] = -f(R[1:-1], Ri[1:-1], rho[1:-1], rhoi[1:-1]) +  Di * np.diff(Ri,2) / dx**2
    dRidt[-1]   = -f(R[-1],   Ri[-1],   rho[-1],   rhoi[-1]) +  Di * (-2.0*Ri[-1] + 2.0*Ri[-2]) / dx**2

    drhodt[0]    = g(R[0],    Ri[0],    rho[0],    rhoi[0]) +  D * (-2.0*rho[0] + 2.0*rho[1]) / dx**2
    drhodt[1:-1] = g(R[1:-1], Ri[1:-1], rho[1:-1], rhoi[1:-1]) +  D * np.diff(rho,2) / dx**2
    drhodt[-1]   = g(R[-1],   Ri[-1],   rho[-1],   rhoi[-1]) +  D * (-2.0*rho[-1] + 2.0*rho[-2]) / dx**2

    drhoidt[0]    = -g(R[0],    Ri[0],    rho[0],    rhoi[0]) +  Di * (-2.0*rhoi[0] + 2.0*rhoi[1]) / dx**2
    drhoidt[1:-1] = -g(R[1:-1], Ri[1:-1], rho[1:-1], rhoi[1:-1]) +  Di * np.diff(rhoi,2) / dx**2
    drhoidt[-1]   = -g(R[-1],   Ri[-1],   rho[-1],   rhoi[-1]) +  Di * (-2.0*rhoi[-1] + 2.0*rhoi[-2]) / dx**2

    dydt[0::4] = dRdt
    dydt[1::4] = dRidt
    dydt[2::4] = drhodt
    dydt[3::4] = drhoidt

    return dydt

# %%
start = time.time()

N = 500
L = 5
x = np.linspace(0, L, N)
dx = L/N

RT = 2
rhoT = 2
D = 0.1
Di = 10

Rss = 0.81747102
Riss = RT - 0.81747102
rhoss = 0.81747102
rhoiss = rhoT - 0.81747102

R0 = Rss*np.ones(np.size(x)) + 0.01*np.sin(2*np.pi*x)
Ri0 = Riss*np.ones(np.size(x))
rho0 = rhoss*np.ones(np.size(x)) - 0.01*np.sin(np.pi*x)
rhoi0 = rhoiss*np.ones(np.size(x))

y0 = np.zeros(4*N)
y0[0::4] = R0
y0[1::4] = Ri0
y0[2::4] = rho0
y0[3::4] = rhoi0

sol = solve_ivp(lambda t,y: rdPDE(t, y, D, Di, dx), [0, 2000], y0, method='LSODA',lband=4,uband=4)

t = sol.t
y = sol.y
# print(t)

# %%

R = y[0::4,:].T
Ri = y[1::4,:].T
rho = y[2::4,:].T
rhoi = y[3::4,:].T
end = time.time()
print(end-start)


# %%
fig = plt.figure("fig1",figsize=(4,3))
ax1 = plt.subplot(111)
ax1.plot(x,R[-1,:],linewidth=2,label=r'$R$')
ax1.plot(x,rho[-1,:],linewidth=2,label=r'$\rho$')
ax1.plot(x,Ri[-1,:],'--',linewidth=2,color='#1f77b4',label=r'$Ri$')
ax1.plot(x,rhoi[-1,:],'--',linewidth=2,color='#ff7f0e',label=r'$\rho_i$')
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax1.set_xlabel(r'$x$')
ax1.set_ylim((0,1.3))
ax1.grid(linewidth=1.5)
# ax1.set_ylabel(r'$')
# ax1.set_title(r'$v$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)
plt.legend(loc=4)

# ax2 = plt.subplot(122)
# pmesh =plt.pcolormesh(x,t,R,cmap=cm.inferno)
# ax2.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
# cbar = fig.colorbar(pmesh,ax=ax2)
# # plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
# cbar.outline.set_linewidth(1.5)
# cbar.ax.tick_params(width=1.5)
# ax2.set_xlabel(r'$x$')
# ax2.set_ylabel(r'$t$')
# ax2.set_title(r'$R$')
# ax2.spines["left"].set_linewidth(1.5)
# ax2.spines["top"].set_linewidth(1.5)
# ax2.spines["right"].set_linewidth(1.5)
# ax2.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('RacRho.tif',dpi=600)
plt.show()
#
# plt.figure()
# print(v[0,:])
# plt.plot(x,v[0,:])

print(R)

print(np.sum(dx*(R[0,:] + Ri[0,:])))
print(np.sum(dx*(R[-1,:] + Ri[-1,:])))
print(np.sum(dx*(rho[0,:] + rhoi[0,:])))
print(np.sum(dx*(rho[-1,:] + rhoi[-1,:])))
Rmass = []
rhomass = []

for i in range(len(t)):
    Rmass.append(np.sum(dx*(R[i,:] + Ri[i,:])))
    rhomass.append(np.sum(dx*(rho[i,:] + rhoi[i,:])))

plt.figure(figsize=(4,3))
plt.plot(Rmass)
plt.plot(rhomass)
plt.show()

# #%%
# # animated plot

movieon = 1
if movieon == 1:
    import matplotlib.animation as animation
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(111)
    ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
    ax.set_xlabel(r'$x$')
    # ax.set_ylabel('Activity')
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    # ax.set_xlim(0,1)
    ax.set_ylim((0,1.3))
    ax.grid(linewidth=1.5)
    # title = plt.title(r'$b$=%1.2f, $\delta$=%1.2f' %(b, delta))
    line_v, = ax.plot(x,R[0,:],linewidth=2,label=r'$R$')
    line_w, = ax.plot(x,rho[0,:],linewidth=2,label=r'$\rho$')
    line_vi, = ax.plot(x,Ri[0,:],'--',linewidth=2,color='#1f77b4',label=r'$Ri$')
    line_wi, = ax.plot(x,rhoi[0,:],'--',linewidth=2,color='#ff7f0e',label=r'$\rho_i$')
    plt.legend(loc=4)
    plt.tight_layout()

    def animate(i):
        line_v.set_ydata(R[i,:])
        line_w.set_ydata(rho[i,:])
        line_vi.set_ydata(Ri[i,:])
        line_wi.set_ydata(rhoi[i,:])
        return line_v, line_w, line_vi, line_wi

    ani = animation.FuncAnimation(fig,animate,frames=np.size(t))
    ani.save("RacRho.mp4",fps=30,dpi=300)
    # ani.save("Schnak.gif",fps=30,writer='imagemagick',dpi=300)
