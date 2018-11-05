### SALib Tutorial - DMDU 2018

SALib is a Python library for global sensitivity analysis (https://github.com/SALib/SALib), which determines the relative influence of model input variables on one or more outputs of interest.

**Installation:** `pip install SALib` 

 **SALib Paper:** [![status](http://joss.theoj.org/papers/431262803744581c1d4b6a95892d3343/status.svg)](http://joss.theoj.org/papers/431262803744581c1d4b6a95892d3343)
 ```
Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis. 
Journal of Open Source Software, 2(9).
```

The library includes multiple methods. For a good review of sensitivity analysis methods, I suggest this paper:
```
Pianosi, Francesca, et al. "Sensitivity analysis of environmental models: A systematic review with practical workflow." Environmental Modelling & Software 79 (2016): 214-232.
```

The tutorial will introduce some theory behind global sensitivity analysis, with a focus on the Sobol method. This will be demonstrated on a simple test function, followed by a discussion of frequently asked questions for interpreting the results.

This tutorial assumes some background in Python programming and statistics, but will be presented at an introductory level.