from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

# function to define model
# in this case it's a test function (Ishigami)
def run_model(x):
  A = 7
  B = 0.1

  y = np.sin(x[0]) + A * np.sin(x[1])**2 + \
      B * x[2]**4 * np.sin(x[0])

  return y

problem = {
  'num_vars': 3,
  'names': ['x1', 'x2', 'x3'],
  'bounds': [[-np.pi, np.pi]]*3
}

# Generate samples
param_values = saltelli.sample(problem, 10000)
N = len(param_values) # number of parameter samples
Y = np.zeros(N)

# Run model for each parameter set, save the output in array Y
for i in range(N):
  Y[i] = run_model(param_values[i])

# Perform sensitivity analysis using the model output
Si = sobol.analyze(problem, Y, print_to_console=True)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)