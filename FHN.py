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


def f(v, w, a, b, epsilon):
    return v * (v - a) * (1 - v) - w

def g(v, w, a, b, epsilon):
    return epsilon * (v - b * w)

def rdPDE(t, y, a, b, epsilon, D, dx):
    """
    The ODEs are derived using the method of lines.
    https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol
    """
    # The vectors v and w are interleaved in y for computational speed
    # every other entry is
    v = y[0::2]
    w = y[1::2]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    dvdt = dydt[::2]
    dwdt = dydt[1::2]

    dvdt[0]    = f(v[0],    w[0],    a, b, epsilon) +  D * (-2.0*v[0] + 2.0*v[1]) / dx**2
    dvdt[1:-1] = f(v[1:-1], w[1:-1], a, b, epsilon) +  D * np.diff(v,2) / dx**2
    dvdt[-1]   = f(v[-1],   w[-1],   a, b, epsilon) +  D * (-2.0*v[-1] + 2.0*v[-2]) / dx**2
    # dvdt[0]    = f(v[0],    w[0],    a, b, epsilon) +  D  * (v[-1] - 2 * v[0] + v[1]) / dx**2
    # dvdt[1:-1] = f(v[1:-1], w[1:-1], a, b, epsilon) +  D * np.diff(v,2) / dx**2
    # dvdt[-1]   = f(v[-1],   w[-1],   a, b, epsilon) +  D * (v[-2] - 2 * v[-1] + v[0] ) / dx**2

    dwdt[0]    = g(v[0],    w[0],    a, b, epsilon)
    dwdt[1:-1] = g(v[1:-1], w[1:-1], a, b, epsilon)
    dwdt[-1]   = g(v[-1],   w[-1],   a, b, epsilon)

    dydt[::2] = dvdt
    dydt[1::2] = dwdt

    return dydt

# %%
# %%time
start = time.time()

N = 1000
L = 1000
x = np.linspace(0, L, N)
dx = L/N

a = -0.1
b = 1e-4
epsilon = 0.005
D = 5

# np.random.seed(42)
# u0 = (a+b)*np.ones(np.size(x)) + 0.01*(2*np.random.rand(np.size(x))-1)
# v0 = (b/(a+b)**2)*np.ones(np.size(x)) + 0.01*(2*np.random.rand(np.size(x))-1)
v0 = np.zeros(np.size(x))
w0 = np.zeros(np.size(x))#0.2*np.exp(-(x+2)**2)
v0[0:10] = 1

y0 = np.zeros(2*N)
y0[::2] = v0
y0[1::2] = w0

sol = solve_ivp(lambda t,y: rdPDE(t, y, a, b, epsilon, D, dx), [0, 2000], y0, t_eval=np.linspace(0,2000,500), method='LSODA',lband=2,uband=2)

t = sol.t
# print(np.shape(t))
y = sol.y
# print(np.shape(y))
# %%

v = y[0::2,:].T
w = y[1::2,:].T

end = time.time()
print(end-start)
# scipy.io.savemat('data.mat',dict(t=t,x=x,u=u,v=v))

# %%
fig = plt.figure("fig1",figsize=(4,3))
ax1 = plt.subplot(111)
pmesh = plt.pcolormesh(x,t,v,cmap=cm.inferno)
cbar = fig.colorbar(pmesh,ax=ax1)
# plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')
ax1.set_title(r'$v$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)

# ax2 = plt.subplot(122)
# pmesh =plt.pcolormesh(x,t,w,cmap=cm.inferno)
# ax2.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
# cbar = fig.colorbar(pmesh,ax=ax2)
# # plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
# cbar.outline.set_linewidth(1.5)
# cbar.ax.tick_params(width=1.5)
# ax2.set_xlabel(r'$x$')
# ax2.set_ylabel(r'$t$')
# ax2.set_title(r'$w$')
# ax2.spines["left"].set_linewidth(1.5)
# ax2.spines["top"].set_linewidth(1.5)
# ax2.spines["right"].set_linewidth(1.5)
# ax2.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('FHN.tif',dpi=600)
# plt.show()
#
# plt.figure()
# print(v[0,:])
# plt.plot(x,v[0,:])

# #%%
# # animated plot


movieon = 0
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
    ax.set_ylim(1.1*np.min([np.min(v),np.min(w)]),1.1*np.max([np.max(v),np.max(w)]))
    ax.grid(linewidth=1.5)
    # title = plt.title(r'$b$=%1.2f, $\delta$=%1.2f' %(b, delta))
    line_v, = ax.plot(x,v[0,:],linewidth=2,label=r'$v$')
    line_w, = ax.plot(x,w[0,:],'--',linewidth=2,label=r'$w$')
    plt.legend(loc=2)
    plt.tight_layout()

    def animate(i):
        line_v.set_ydata(v[i,:])
        line_w.set_ydata(w[i,:])
        return line_v, line_w

    ani = animation.FuncAnimation(fig,animate,frames=np.size(t))
    ani.save("FHN.mp4",fps=30,dpi=300)
    # ani.save("Schnak.gif",fps=30,writer='imagemagick',dpi=300)
