# fourier bases in all directions periodic bcs
# this one works

import dedalus.public as d3
import matplotlib.pylab as plt 
from matplotlib.pylab import seed
import numpy as np
import time

total_time = 5.0
dt = 0.001
Re = 1000.0

Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2*np.pi
Nx = 32
Ny = 32
Nz = 32

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

U0 = 1
u['g'][:] = U0
v['g'][:] = 0
w['g'][:] = 0
p['g'][:] = 0

#---------------------------------------------------------------------
# apparently this is not good 
# Add small random perturbations to the velocity fields
# np.random.seed(42)
# epsilon = 0.01 * U0 
# u['g'] += epsilon * np.random.randn(*u['g'].shape)
# v['g'] += epsilon * np.random.randn(*v['g'].shape)
# w['g'] += epsilon * np.random.randn(*w['g'].shape)

#---------------------------------------------------------------------
# potential ou initial forcing to initiate turbulence

# def forcing(u,v,w, kmax=3, sigma=0.5, seed=123):
#     rng = np.random.default_rng(seed)
#     uc = u['c'].copy()
#     vc = v['c'].copy()
#     wc = w['c'].copy()

#     kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
#     ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
#     kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)

#     KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
#     K2 = KX**2 + KY**2 + KZ**2
#     mask = (K2 > 0) & (np.sqrt(K2) <= kmax)

#     fx = sigma * rng.normal(size=uc.shape) * 1j*rng.normal(size=uc.shape)
#     fy = sigma * rng.normal(size=uc.shape) * 1j*rng.normal(size=uc.shape)
#     fz = sigma * rng.normal(size=uc.shape) * 1j*rng.normal(size=uc.shape) 

#     fx *= mask
#     fy *= mask
#     fz *= mask

#     invK2 = np.zeros_like(K2)
#     invK2[K2 > 0] = 1 / K2[K2 > 0]
#     kf = KX * fx + KY * fy + KZ * fz
#     fx -= KX * kf * invK2
#     fy -= KY * kf * invK2
#     fz -= KZ * kf * invK2

#     u["c"][:] += fx
#     v["c"][:] += fy
#     w["c"][:] += fz

# forcing(u,v,w, kmax=3, sigma=0.5, seed=123)
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

    return{"sigma_xx": Sxx_g[ix,iy,iz], "sigma_yy": Syy_g[ix,iy,iz], "sigma_zz": Szz_g[ix,iy,iz],
           "sigma_xy": Sxy_g[ix,iy,iz], "sigma_xz": Sxz_g[ix,iy,iz], "sigma_yz": Syz_g[ix,iy,iz], "S_mag": S_mag} 

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
    solver.step(dt)
    t = solver.sim_time

    # this bit uses the stress calc
    stress = stress_calc(u,v,w,p,Re, xp, yp, zp)
    time_list.append(t)
    sigma_xx_list.append(stress["sigma_xx"])
    sigma_yy_list.append(stress["sigma_yy"])
    sigma_zz_list.append(stress["sigma_zz"])
    sigma_xy_list.append(stress["sigma_xy"])
    sigma_xz_list.append(stress["sigma_xz"])
    sigma_yz_list.append(stress["sigma_yz"])
    S_mag_list.append(stress["S_mag"])

    if solver.iteration % 100 == 0:
        print(f"Iteration: {solver.iteration}, Time: {t:.3f}")
    
end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

# plot pls work
plt.figure(figsize=(10, 6))
plt.plot(time_list, sigma_xx_list, label='Sigma_xx')
plt.plot(time_list, sigma_yy_list, label='Sigma_yy')
plt.plot(time_list, sigma_zz_list, label='Sigma_zz')
plt.plot(time_list, sigma_xy_list, label='Sigma_xy')
plt.plot(time_list, sigma_xz_list, label='Sigma_xz')
plt.plot(time_list, sigma_yz_list, label='Sigma_yz')
plt.plot(time_list, S_mag_list, label='S_mag')
plt.xlabel('Time')
plt.ylabel('Stress Components / Strain Rate Magnitude')
plt.title('Stress Components and Strain Rate Magnitude Over Time at Point (%.2f, %.2f, %.2f)' % (xp, yp, zp))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()