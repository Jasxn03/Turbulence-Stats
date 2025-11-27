import dedalus.public as d3
import numpy as np

total_time = 1.0
dt = 0.00001
Re = 1000.0

Lx = 2*np.pi
Ly = 2*np.pi
Lz = 2*np.pi
Nx = 32
Ny = 32
Nz = 32

tau_sx = 0.001     # nondimensional wind stress in x
tau_sy = 0.001     # nondimensional wind stress in y

xcoord = d3.Coordinate('x') 
ycoord = d3.Coordinate('y')
zcoord = d3.Coordinate('z')
dist = d3.Distributor((xcoord, ycoord, zcoord), dtype=np.float64)
xbasis = d3.Fourier(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2, dtype=np.float64)
ybasis = d3.Fourier(ycoord, size=Ny, bounds=(0, Ly), dealias=3/2, dtype=np.float64)
zbasis = d3.Chebyshev(zcoord, size=Nz, bounds=(0, Lz), dealias=3/2) #d3.Fourier enforces periodc bcs 

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

tau_1 = dist.Field(name='tau_1', bases=[xbasis, ybasis]) 
tau_2 = dist.Field(name='tau_2', bases=(xbasis,ybasis))
tau_3 = dist.Field(name='tau_3', bases=(xbasis,ybasis)) 
tau_4 = dist.Field(name='tau_4', bases=[xbasis, ybasis]) 
tau_5 = dist.Field(name='tau_5', bases=[xbasis, ybasis]) 
tau_6 = dist.Field(name='tau_6', bases=[xbasis, ybasis]) # Tau field for pressure at the top boundary])
tau_p = dist.Field(name='tau_p', bases=None)
lift_basis = zbasis.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

problem = d3.IVP([u, v, w, p, tau_1, tau_2, tau_3, tau_4, tau_5, tau_6, tau_p], namespace={"Re":Re, "dx":dx, "dy":dy, "dz":dz, "div":div, "lap":lap, "lift_basis":lift_basis, "lift":lift, 'Lz':Lz, 'tau_sx':tau_sx, 'tau_sy':tau_sy})
# problem.add_equation("dt(u) - (1/Re)*lap(u) + dx(p) + lift(tau_1) = -u*dx(u) - v*dy(u) - w*dz(u) + w*lift(tau_4) - (1/Re)*dz(lift(tau_4)) ")
# problem.add_equation("dt(v) - (1/Re)*lap(v) + dy(p) + lift(tau_2) = -u*dx(v) - v*dy(v) - w*dz(v) + w*lift(tau_5) - (1/Re)*dz(lift(tau_5))")
# problem.add_equation("dt(w) - (1/Re)*lap(w) + dz(p) + lift(tau_3) = -u*dx(w) - v*dy(w) - w*dz(w) + w*lift(tau_6) - (1/Re)*dz(lift(tau_6))")

problem.add_equation("dt(u) - (1/Re)*lap(u) + dx(p) + lift(tau_1) = -u*dx(u) - v*dy(u) - w*dz(u) ")
problem.add_equation("dt(v) - (1/Re)*lap(v) + dy(p) + lift(tau_3)  = -u*dx(v) - v*dy(v) - w*dz(v)")
problem.add_equation("dt(w) - (1/Re)*lap(w) + dz(p) + lift(tau_5)  = -u*dx(w) - v*dy(w) - w*dz(w)")

problem.add_equation("div(u, v, w) + tau_p = 0")
problem.add_equation("integ(p) = 0")

# Boundary conditions in z as i have used chebyshev basis:
problem.add_equation("u(z=0) = 0")
# problem.add_equation("u(z=Lz) = 0")
problem.add_equation("v(z=0) = 0")
# problem.add_equation("v(z=Lz) = 0") 
problem.add_equation("w(z=0) = 0")
# problem.add_equation("w(z=Lz) = 0")
problem.add_equation("dz(u)(z=Lz) =  + tau_sx/Re ") # Stress BC at top
problem.add_equation("dz(v)(z=Lz) =  + tau_sy/Re ") # Stress BC at top
problem.add_equation("w(z=Lz) = 0") # No normal flow at the top boundary

U0 = 1
u['g'][:] = U0
v['g'][:] = 0
w['g'][:] = 0
p['g'][:] = 0

# Add small random perturbations to the velocity fields
np.random.seed(42)
epsilon = 0.01 * U0 
u['g'] += epsilon * np.random.randn(*u['g'].shape)
v['g'] += epsilon * np.random.randn(*v['g'].shape)
w['g'] += epsilon * np.random.randn(*w['g'].shape)

solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = total_time

while solver.proceed:
    solver.step(dt)
    t = solver.sim_time
    if solver.iteration % 100 == 0:
        print(f"Iteration: {solver.iteration}, Time: {t:.3f}")

