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
orders_data = {i: {j: group.at[(i,j), 'Quantity'] for j in group.index.get_level_values(1)}
                                                    for i, group in orders_df.groupby(level=0)}

# {product ID: weight}
product_weights = {p: product_weight_df.at[p, 'Weight'] for p in product_weight_df.index}

# {warehouse ID: {order ID: cost}}
costs = costs_df.set_index('Warehouse ID/OrderID').stack()
costs = {i: {int(j[1]): group.at[j] for j in group.index} for i, group in costs.groupby(level=0)}

# {warehouse ID: {product ID: stock}}
warehouse_df = warehouse_df.set_index(['Warehouse ID', 'Product ID'])
warehouse_stock = {i: {j: group.at[(i,j), 'Stock'] for j in group.index.get_level_values(1)}
                                                    for i, group in warehouse_df.groupby(level=0)}

# Make some useful lists
warehouses = list(warehouse_stock)
orders = list(orders_data)
products = list(product_weights)

#instantiate model object
m = gp.Model('Model A')

## DECISION VARIABLES
flow = {}

# index is (warehouse, order, product)
indices = [(i, j, k) for i in warehouses for j in orders for k in products if k in orders_data[j]]
flow = m.addVars(indices, lb=0, vtype=GRB.CONTINUOUS)

# create delta variables for orders that cannot be satisfied and leftover supply
order_delta_indices = [(o, k) for o in orders for k in products if k in orders_data[o]]
supply_delta_indices = [(w, k) for w in warehouses for k in products]
order_delta = m.addVars(order_delta_indices, lb=0, vtype=GRB.CONTINUOUS)
supply_delta = m.addVars(supply_delta_indices, lb=0, vtype=GRB.CONTINUOUS)
tot_unfulfilled_demand = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
tot_leftover_supply = m.addVar(lb=0, vtype=GRB.CONTINUOUS)

# create cost variable for delivery
cost = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='total_cost')

## OBJECTIVE FUNCTION
#heavily penalize delta total so that orders are as satisfied when possible
m.setObjective(cost + 10000 * (tot_unfulfilled_demand+tot_leftover_supply), sense=GRB.MINIMIZE)

## CONSTRAINTS
# Define cost variable
m.addConstr(cost == sum(flow[t]*costs[t[0]][t[1]]*product_weights[t[2]] for t in indices))

# meet demand
for o in orders:
    for k in orders_data[o]:
        m.addConstr(sum(flow[(w, o, k)] for w in warehouses) + order_delta[(o, k)] == orders_data[o][k])

# don't exceed stock
for w in warehouses:
    for k in products:
        m.addConstr(sum(flow[(w, o, k)] for o in orders if k in orders_data[o]) + supply_delta[(w,k)]
                            == warehouse_stock[w][k])

# define order delta total
m.addConstr(tot_unfulfilled_demand == sum(order_delta[t] for t in order_delta_indices))
m.addConstr(tot_leftover_supply == sum(supply_delta[t] for t in supply_delta_indices))

# Solve model
m.optimize()
status_code =   {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
status = m.status
print(f'The optimization status is {status_code[status]}')

# print solution if optimal
if status == 2:
    print('Warehouse\t Order\t Product\t Quantity\t')
    for t in indices:
        if flow[t].X != 0:
            print(f'{t[0]}\t {t[1]}\t {t[2]}\t {flow[t].X}')
    print(f'Obj function value: {m.objVal}')
    print(f'Total order cost: {cost.X}')
    print('-------')
    print(f'Total unfulfilled demand: {tot_unfulfilled_demand.X} units')
    print(f'Order\t Product\t Unfulfilled Demand')
    for o in orders:
        for k in orders_data[o]:
            if order_delta[(o,k)].X != 0:
                print(f'{o}\t {k}\t {order_delta[(o,k)].X}')
    print('-------')
    print(f'Total leftover supply: {tot_leftover_supply.X} units')
    print(f'Warehouse\t Product\t Leftover Supply')
    for w in warehouses:
        for k in warehouse_stock[w]:
            if supply_delta[(w,k)].X != 0:
                print(f'{w}\t {k}\t {supply_delta[(w,k)].X}')


