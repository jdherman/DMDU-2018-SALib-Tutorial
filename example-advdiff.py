import numpy as np
from scipy import integrate
from SALib.sample import saltelli
from SALib.analyze import sobol

# model set up as a function. Accepts an array of parameter values, x.
def advdiff(x):
  theta = x[0]     # [-] Effective porosity
  R = x[1]         # [-] Retardation factor
  M = x[2]         # [g] Mass of solute
  v = 10**x[3]         # [m/d] Seepage velocity
  Dstar = 10**x[4]     # [m^2/d] Molecular diffusion coefficient
  t_half = x[5]    # [d] half life

  #  Don't change these
  Dx = 0.36*v + Dstar  # [m^2/d] Longitudinal dispersivity
  Dy = 0.036*v + Dstar # [m^2/d] Transverse dispersivity
  Dz = 0.036*v + Dstar # [m^2/d] Transverse dispersivity
  x = 10 
  y = 5 
  z = -4.5 # [m] Location of well

  # Dimensionless system parameters
  Mp = M/(theta*R)       # Mass
  Dxp = Dx/R           # L Dispersion
  Dyp = Dy/R           # T Dispersion
  Dzp = Dz/R           # T Dispersion
  vp = v/R             # Velocity
  l = np.log(2)/t_half # Reaction rate

  fun = lambda t,j: (t**j*((Mp/(8*(np.pi*t)**(3/2) * np.sqrt(Dxp*Dyp*Dzp))) *
      np.exp(-(x-vp*t)**2/(4*Dxp*t) - (y**2)/(4*Dyp*t) - (z**2)/(4*Dzp*t) - l*t)))

  # absolute moments
  m0t = integrate.quad(fun, 0, np.inf, args=(0))[0]  # zeroth
  m1t = integrate.quad(fun, 0, np.inf, args=(1))[0]  # first

  # do not allow division by zero
  if m0t < 1e-35:
    m0t = 1e-35

  # normalized absolute moments
  mu0tp = m0t/m0t # zeroth
  mu1tp = m1t/m0t # first

  return m0t #[m0t, mu1tp]


problem = {
  'num_vars': 6,
  'names': ['theta', 'R', 'M', 'v', 'Dstar', 't_half'],
  'bounds': [[0.03, 0.45],
             [2.0, 10.0],
             [9000, 11000],
             [-3, -1],
             [-6, -4],
             [5000, 10000]]
}

# Generate samples
param_values = saltelli.sample(problem, 1000)
N = len(param_values) # number of parameter samples
Y = np.zeros(N)

# Run model for each parameter set, save the output in array Y
for i in range(N):
  if i % 1000 == 0:
    print(i)

  Y[i] = advdiff(param_values[i])

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)
# # Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# # (first and total-order indices with bootstrap confidence intervals)