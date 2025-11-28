# fourier bases in all directions periodic bcs
# this one works
#
# can i use other initial conditions? something that is more realistic?
# stochastic forcing is fine, ewaran (?) and pope 1987/88
# can get stress field by computing all the points in? then i sorta have a 3d heat map sorta situation?
#
# MY GRID BOX IS 32x32x32 IS DISCRETIZED BY 32 POINTS - 2PI/32 = 0.2 this takes 4mins for t = 3 and dt = 0.001
#
# t=50 and ou forcing turbulence, the vortex gets pushed away? this shows transport possibily?


import dedalus.public as d3
import matplotlib.pylab as plt 
from matplotlib.pylab import seed
import numpy as np
import time

total_time = 15.0
dt = 0.001
Re = 1000.0

Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2*np.pi
Nx = 128
Ny = 128
Nz = 128

xcoord = d3.Coordinate('x') 
ycoord = d3.Coordinate('y')
zcoord = d3.Coordinate('z')
dist = d3.Distributor((xcoord, ycoord, zcoord), dtype=np.float64)
xbasis = d3.Fourier(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2, dtype=np.float64)
ybasis = d3.Fourier(ycoord, size=Ny, bounds=(0, Ly), dealias=3/2, dtype=np.float64)
zbasis = d3.Fourier(zcoord, size=Nz, bounds=(0, Lz), dealias=3/2, dtype=np.float64) #d3.Fourier enforces periodc bcs 

u = dist.Field(name='u', bases=(xbasis, ybasis, zbasis))
v = dist.Field(name='v', bases=(xbasis, ybasis, zbasis))
w = dist.Field(name='w', bases=(xbasis, ybasis, zbasis))
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))

dx = lambda A: d3.Differentiate(A, xcoord)
dy = lambda A: d3.Differentiate(A, ycoord)
dz = lambda A: d3.Differentiate(A, zcoord)
div = lambda A, B, C: dx(A) + dy(B) + dz(C)
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
grad = lambda A: (dx(A), dy(A), dz(A))

tau_p = dist.Field(name='tau_p', bases=None)

problem = d3.IVP([u, v, w, p, tau_p], namespace={"Re":Re, "dx":dx, "dy":dy, "dz":dz, "div":div, "lap":lap, 'Lz':Lz})

problem.add_equation("dt(u) - (1/Re)*lap(u) + dx(p) = -u*dx(u) - v*dy(u) - w*dz(u)")
problem.add_equation("dt(v) - (1/Re)*lap(v) + dy(p) = -u*dx(v) - v*dy(v) - w*dz(v)")
problem.add_equation("dt(w) - (1/Re)*lap(w) + dz(p) = -u*dx(w) - v*dy(w) - w*dz(w)")
problem.add_equation("div(u, v, w) + tau_p = 0")
problem.add_equation("integ(p) = 0")

#---------------------------------------------------------------------

# taylor green vortex ic  for turbulence
# Add small random perturbations to the velocity fields
# U0 = 1
# kx = ky = kz = 1
# x = dist.local_grids(xbasis, ybasis, zbasis)[0]
# y = dist.local_grids(xbasis, ybasis, zbasis)[1]
# z = dist.local_grids(xbasis, ybasis, zbasis)[2]
# u['g'][:] = U0 * np.sin(kx*x) * np.cos(ky*y)*np.cos(kz*z)
# v['g'][:] = -U0 * np.cos(kx*x)* np.sin(ky*y) * np.cos(kz*z)
# w['g'][:] = 0
# p['g'][:] = 0
# can keep the small pertubation after if want to
#---------------------------------------------------------------------

# apparently this is not good 
# Add small random perturbations to the velocity fields
U0 = 1
u['g'][:] = U0
v['g'][:] = 0
w['g'][:] = 0
p['g'][:] = 0
# np.random.seed(42)
# epsilon = 0.01 * U0 
# u['g'] += epsilon * np.random.randn(*u['g'].shape)
# v['g'] += epsilon * np.random.randn(*v['g'].shape)
# w['g'] += epsilon * np.random.randn(*w['g'].shape)

#---------------------------------------------------------------------

# potential ou initial forcing to initiate turbulence

rng = np.random.default_rng(123)

def forcing(u,v,w, kmax=5, tau = 0.5, sigma=0.5):
    
    shape = u['g'].shape
    du = np.zeros(shape)
    dv = np.zeros(shape)
    dw = np.zeros(shape)

    Nx, Ny, Nz = shape

    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    mask = (K2 > 0) & (np.sqrt(K2) <= kmax)

    du = sigma * rng.standard_normal(shape) * mask
    dv = sigma * rng.standard_normal(shape) * mask
    dw = sigma * rng.standard_normal(shape) * mask

    alpha = np.exp(-dt/tau)
    beta = np.sqrt(1-alpha**2)

    u['g'][:] = alpha*u['g'] + beta*du
    v['g'][:] = alpha*v['g'] + beta*dv
    w['g'][:] = alpha*w['g'] + beta*dw

    u.require_coeff_space()
    v.require_coeff_space()
    w.require_coeff_space()

forcing(u,v,w, kmax=3, sigma=0.5)

#---------------------------------------------------------------------

# now scary stress stuff that hopefully works

def stress_calc(u,v,w,p,Re, xp, yp, zp):

    Sxx = dx(u)
    Syy = dy(v)
    Szz = dz(w)
    Sxy = 0.5*(dy(u) + dx(v))
    Sxz = 0.5*(dz(u) + dx(w))
    Syz = 0.5*(dz(v) + dy(w))   

    sigma_xx = -p + (2/Re)*Sxx
    sigma_yy = -p + (2/Re)*Syy
    sigma_zz = -p + (2/Re)*Szz
    sigma_xy = (2/Re)*Sxy
    sigma_xz = (2/Re)*Sxz
    sigma_yz = (2/Re)*Syz

    Sxx_g = sigma_xx.evaluate()['g']
    Syy_g = sigma_yy.evaluate()['g']
    Szz_g = sigma_zz.evaluate()['g']
    Sxy_g = sigma_xy.evaluate()['g']
    Sxz_g = sigma_xz.evaluate()['g']
    Syz_g = sigma_yz.evaluate()['g']

    ix = int(xp/Lx * Nx) % Nx
    iy = int(yp/Ly * Ny) % Ny
    iz = int(zp/Lz * Nz) % Nz

    S_mag = np.sqrt(Sxx_g[ix,iy,iz]**2 + Syy_g[ix,iy,iz]**2 + Szz_g[ix,iy,iz]**2 +
                    2*(Sxy_g[ix,iy,iz]**2 + Sxz_g[ix,iy,iz]**2 + Syz_g[ix,iy,iz]**2))

    return{"S_mag": S_mag} 

xp = Lx/2
yp = Ly/2
zp = Lz/2

time_list = []
sigma_xx_list = []
sigma_yy_list = []
sigma_zz_list = []
sigma_xy_list = []
sigma_xz_list = []
sigma_yz_list = []
S_mag_list = []

#---------------------------------------------------------------------

start_time = time.time()

solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = total_time

while solver.proceed:

    #potential ou forcing here
    forcing(u,v,w, kmax=3, tau = 0.5, sigma=0.5)

    solver.step(dt)
    t = solver.sim_time

    # this bit uses the stress calc
    stress = stress_calc(u,v,w,p,Re, xp, yp, zp)
    time_list.append(t)
    S_mag_list.append(stress["S_mag"])

    if solver.iteration % 100 == 0:
        print(f"Iteration: {solver.iteration}, Time: {t:.3f}")
    
end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

print("Max S_mag:", np.max(stress_calc(u,v,w,p,Re,xp,yp,zp)))

# plot pls work
plt.figure(figsize=(10, 6))
plt.plot(time_list, S_mag_list, label='S_mag')
plt.xlabel('Time')
plt.ylabel('Strain Rate Magnitude')
plt.title('Strain Rate Magnitude Over Time at Point (%.2f, %.2f, %.2f)' % (xp, yp, zp))
plt.legend()
plt.show()

#---------------------------------------------------------------------

from mpl_toolkits.mplot3d import Axes3D

xg, yg, zg = dist.local_grids(xbasis, ybasis, zbasis)
Nx_local, Ny_local, Nz_local = u['g'].shape

x1d = np.linspace(0, Lx, Nx_local, endpoint=False)
y1d = np.linspace(0, Ly, Ny_local, endpoint=False)
z1d = np.linspace(0, Lz, Nz_local, endpoint=False)

x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')

slice_z_idx = Nz_local // 2

# Extract slices
u_slice = u['g'][:, :, slice_z_idx]
v_slice = v['g'][:, :, slice_z_idx]
w_slice = w['g'][:, :, slice_z_idx]

u2 = u_slice.T
v2 = v_slice.T

plt.figure(figsize=(8,6))
plt.contourf(x2d, y2d, u_slice, 40, cmap='RdBu_r')
plt.colorbar(label="u velocity")
plt.xlabel("x"); plt.ylabel("y")
plt.title("u velocity slice")
plt.show()

plt.figure(figsize=(8,6))
plt.streamplot(x1d, y1d, u2, v2, density=1.4, linewidth=1)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Streamlines (u,v)")
plt.show()

plt.figure(figsize=(8,6))
plt.contourf(x2d, y2d, w_slice, 40, cmap='coolwarm')
plt.streamplot(x1d, y1d, u2, v2, color='k', density=1.3)
plt.colorbar(label='w velocity')
plt.xlabel("x"); plt.ylabel("y")
plt.title("w + streamlines of (u,v)")
plt.show()

x = dist.local_grids(xbasis, ybasis, zbasis)[0]
y = dist.local_grids(xbasis, ybasis, zbasis)[1]
z = dist.local_grids(xbasis, ybasis, zbasis)[2]

u.change_scales(1)
v.change_scales(1)
w.change_scales(1)

u_grid = u['g']
v_grid = v['g']
w_grid = w['g']

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

xs = X.ravel()
ys = Y.ravel()
zs = Z.ravel()
ws = w_grid.ravel() 

print(xs.shape, ys.shape, zs.shape, ws.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# xs, ys, zs = arrays all of length N
# ws = array of same length N (your value to color by)

p = ax.scatter(xs, ys, zs, c=ws, cmap='coolwarm', s=8)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

fig.colorbar(p, ax=ax, label="w value")
plt.show()



