#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:15:26 2020
@student: p williams
A = [[9,6],[8,3],[7,4]]
"""

from gurobipy import GRB,Model

#Week 6 Quiz Example
# Create the model
m = Model('q6')
# Set parameters
m.setParam('OutputFlag',True)
# Add variables
x1 = m.addVar(name='x1')
x2 = m.addVar(name='x2')
# Add constraints
m.addConstr(9*x1 +  6*x2 <= 3, name='c1')
m.addConstr(8*x1 +  3*x2 <= 14, name='c2')
m.addConstr(7*x1 +  4*x2 <= 10, name='c3')
#Set the objective
m.setObjective(x1 - 2*x2, GRB.MAXIMIZE)
# Optimize the model
m.optimize()
# Print the result
status_code =   {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
status = m.status
print('The optimization status is {}'.format(status_code[status]))

if status == 2:
    # Retrieve variables value
    print('Optimal solution:')
    for v in m.getVars():
        print('%s = %g' % (v.varName, v.x))
        print('Optimal objective value:  \n{}'.format(m.objVal))