import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io

plt.rc("text", usetex=False)
plt.rc("font", family="sans-serif", size=12)


def f(u, v, a, b, gamma):
    return gamma*(a-u+u**2*v)

def g(u, v, a, b, gamma):
    return gamma*(b-u**2*v)

def rdPDE(y, t, a, b, gamma, Du, Dv, dx):
    """
    The ODEs are derived using the method of lines.
    https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol
    """
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    u = y[::2]
    v = y[1::2]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    dudt = dydt[::2]
    dvdt = dydt[1::2]

    dudt[0]    = f(u[0],    v[0],    a, b, gamma) +  Du  * (-2.0*u[0] + 2.0*u[1]) / dx**2
    dudt[1:-1] = f(u[1:-1], v[1:-1], a, b, gamma) +  Du * np.diff(u,2) / dx**2
    dudt[-1]   = f(u[-1],   v[-1],   a, b, gamma) +  Du * (-2.0*u[-1] + 2.0*u[-2]) / dx**2

    dvdt[0]    = g(u[0],    v[0],    a, b, gamma) +  Dv * (-2.0*v[0] + 2.0*v[1]) / dx**2
    dvdt[1:-1] = g(u[1:-1], v[1:-1], a, b, gamma) +  Dv * np.diff(v,2) / dx**2
    dvdt[-1]   = g(u[-1],   v[-1],   a, b, gamma) +  Dv * (-2.0*v[-1] + 2.0*v[-2]) / dx**2

    dydt[::2] = dudt
    dydt[1::2] = dvdt

    return dydt

# %%
# %%time
N = 100
x = np.linspace(0, 1, N)
dx = 1/N
T = 1
M = 100
t = np.linspace(0, T, M)

a, b = 0.1, 0.9
gamma = 400
Du = 1
Dv = 10

np.random.seed(42)
u0 = (a+b)*np.ones(np.size(x)) + 0.01*(2*np.random.rand(np.size(x))-1)
v0 = (b/(a+b)**2)*np.ones(np.size(x)) + 0.01*(2*np.random.rand(np.size(x))-1)

y0 = np.zeros(2*N)
y0[::2] = u0
y0[1::2] = v0


sol = odeint(rdPDE, y0, t, args=(a, b, gamma, Du, Dv, dx), ml=2, mu=2)

# %%

u = sol[:,0::2]
v = sol[:,1::2]

# scipy.io.savemat('data.mat',dict(t=t,x=x,u=u,v=v))

# %%
fig = plt.figure("fig1",figsize=(5.2,2.6))
ax1 = plt.subplot(121)
pmesh = plt.pcolormesh(x,t,u,cmap=cm.inferno)
cbar = fig.colorbar(pmesh,ax=ax1)
# plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')
ax1.set_title(r'$u$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)

ax2 = plt.subplot(122)
pmesh =plt.pcolormesh(x,t,v,cmap=cm.inferno)
ax2.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
cbar = fig.colorbar(pmesh,ax=ax2)
# plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$t$')
ax2.set_title(r'$v$')
ax2.spines["left"].set_linewidth(1.5)
ax2.spines["top"].set_linewidth(1.5)
ax2.spines["right"].set_linewidth(1.5)
ax2.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
# plt.savefig('case1_fixed.png',dpi=600)
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
    ax.set_xlim(0,1)
    ax.set_ylim(0.9*np.min([np.min(u),np.min(v)]),1.1*np.max([np.max(u),np.max(v)]))
    ax.grid(linewidth=1.5)
    # title = plt.title(r'$b$=%1.2f, $\delta$=%1.2f' %(b, delta))
    line_u, = ax.plot(x,u[0,:],linewidth=2,label=r'$u$')
    line_v, = ax.plot(x,v[0,:],'--',linewidth=2,label=r'$v$')
    plt.legend(loc=2)
    plt.tight_layout()

    def animate(i):
        line_u.set_ydata(u[i,:])
        line_v.set_ydata(v[i,:])
        return line_u, line_v

    ani = animation.FuncAnimation(fig,animate,frames=M)
    ani.save("Schnak.mp4",fps=30,dpi=300)
    # ani.save("Schnak.gif",fps=30,writer='imagemagick',dpi=300)
