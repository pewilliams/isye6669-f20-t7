import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# read in necessary data into dataframes
orders_df = pd.read_csv('../csv/orders.csv')
product_weight_df = pd.read_csv('../csv/ProductWeight.csv').set_index('Product ID')
warehouse_df = pd.read_csv('../csv/Warehouses.csv')
costs_df = pd.read_csv('../csv/DeliveryCost.csv')

#turn dataframes into dictionaries

# {order id: product id: amount} only for product ids with amount > 0
orders_df = orders_df.groupby(['Order ID', 'Product ID']).sum()
orders_data = {i: {j: group.at[(i,j), 'Quantity'] for j in group.index.get_level_values(1)} for i, group in orders_df.groupby(level=0)}

# {product ID: weight}
product_weights = {p: product_weight_df.at[p, 'Weight'] for p in product_weight_df.index}

# {warehouse ID: {product ID: cost}}
costs = costs_df.set_index('Warehouse ID/OrderID').stack()
costs = {i: {int(j[1]): group.at[j] for j in group.index} for i, group in costs.groupby(level=0)}

# {warehouse ID: {product ID: stock}}
warehouse_df = warehouse_df.set_index(['Warehouse ID', 'Product ID'])
warehouse_stock = {i: {j: group.at[(i,j), 'Stock'] for j in group.index.get_level_values(1)} for i, group in warehouse_df.groupby(level=0)}

# Make some useful lists
warehouses = list(warehouse_stock)
orders = list(orders_data)
products = list(product_weights)

#instantiate model object
m = gp.Model('Model A')

# DECISION VARIABLES
flow = {}

# index is (warehouse, order, product)
indices = [(i, j, k) for i in warehouses for j in orders for k in products if k in orders_data[j]]
flow = m.addVars(indices, lb=0, vtype=GRB.CONTINUOUS)

# create delta variables for orders that cannot be satisfied
deltas_indices = [(o, k) for o in orders for k in products if k in orders_data[o]]
order_delta = m.addVars(deltas_indices, lb=0, vtype=GRB.CONTINUOUS)
delta_total = m.addVar(lb=0, vtype=GRB.CONTINUOUS)

# create cost variable for delivery
cost = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='total_cost')

#heavily penalize delta total so that orders are as satisfied when possible
m.setObjective(cost + 10000*delta_total, sense=GRB.MINIMIZE)

# Define cost variable
m.addConstr(cost == sum(flow[t]*costs[t[0]][t[1]] for t in indices))

# meet demand
for o in orders:
    for k in orders_data[o]:
        m.addConstr(sum(flow[(w, o, k)] for w in warehouses) == orders_data[o][k])

# don't exceed stock
for w in warehouses:
    for k in products:
        m.addConstr(sum(flow[(w, o, k)]-order_delta[(o, k)] for o in orders if k in orders_data[o]) <= warehouse_stock[w][k])

# define order delta total
m.addConstr(delta_total == sum(order_delta[t] for t in deltas_indices))
m.optimize()

for t in indices:
    if flow[t].X != 0:
        print(t, flow[t].X)
print(f'Obj: {m.objVal}')
print(f'Cost: {cost.X}')




