import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# read in necessary data into dataframes
orders_df = pd.read_csv('../csv/orders.csv')
product_weight_df = pd.read_csv('../csv/ProductWeight.csv').set_index('Product ID')
warehouse_df = pd.read_csv('../csv/Warehouses.csv')
costs_df = pd.read_csv('../csv/DeliveryCost.csv')
fixed_costs_df = pd.read_csv('../csv/FixedCosts.csv')
warehouse_regions_df = pd.read_csv('../csv/WarehouseRegions.csv')
order_regions_df = pd.read_csv('../csv/OrderRegions.csv')
#turn dataframes into dictionaries

# {order id: product id: amount} only for product ids with amount > 0
orders_df = orders_df.groupby(['Order ID', 'Product ID']).sum()
orders_data = {i: {j: group.at[(i,j), 'Quantity'] for j in group.index.get_level_values(1)}
                                                    for i, group in orders_df.groupby(level=0)}

# {product ID: weight}
product_weights = {p: product_weight_df.at[p, 'Weight'] for p in product_weight_df.index}

# {warehouse ID: {order ID: unit cost}}
costs = costs_df.set_index('Warehouse ID/OrderID').stack()
costs = {i: {int(j[1]): group.at[j] for j in group.index} for i, group in costs.groupby(level=0)}

# {warehouse ID: {order ID: fixed cost}
fixed_costs = fixed_costs_df.set_index('Warehouse ID/Order ID').stack()
fixed_costs = {i: {int(j[1]): group.at[j] for j in group.index} for i, group in fixed_costs.groupby(level=0)}

# {warehouse ID: {product ID: stock}}
warehouse_df = warehouse_df.set_index(['Warehouse ID', 'Product ID'])
warehouse_stock = {i: {j: group.at[(i,j), 'Stock'] for j in group.index.get_level_values(1)}
                                                    for i, group in warehouse_df.groupby(level=0)}
# {warehouse ID: region}
warehouse_regions = {warehouse_regions_df.at[i, 'Warehouse ID']: warehouse_regions_df.at[i, 'Region']
                     for i in warehouse_regions_df.index}

# {order ID: region}
order_regions = {order_regions_df.at[i, 'Order ID']: order_regions_df.at[i, 'Region']
                    for i in order_regions_df.index}

# Make some useful lists
warehouses = list(warehouse_stock)
orders = list(orders_data)
products = list(product_weights)
regions = list(set(order_regions.values()))

# get stock for each product type in each region
region_stock = {r: {k: sum([warehouse_stock[w][k] for w in warehouses if warehouse_regions[w]==r]) for k in products}
                for r in regions}

#instantiate model object
m = gp.Model('Model A')

## DECISION VARIABLES
flow = {}

# index is (warehouse, order, product)
indices = [(i, j, k) for i in warehouses for j in orders for k in products if k in orders_data[j]]
flow = m.addVars(indices, lb=0, vtype=GRB.INTEGER)

#binary variable for incurring fixed cost
fixed_cost_ind = m.addVars([(w,o) for w in warehouses for o in orders], vtype=GRB.BINARY)
# use max stock as big M to activate fixed costs ind
max_stock = max(warehouse_stock[w][k] for w in warehouses for k in products)
# use max regional demand for big M in nonregional flow constraint
regional_demand = {r: {k: sum(orders_data[o][k] for o in orders if k in orders_data[o]) for k in products}
                    for r in regions}

# create delta variables for orders that cannot be satisfied and leftover supply
order_delta_indices = [(o, k) for o in orders for k in products if k in orders_data[o]]
supply_delta_indices = [(w, k) for w in warehouses for k in products]
order_delta = m.addVars(order_delta_indices, lb=0, vtype=GRB.CONTINUOUS)
supply_delta = m.addVars(supply_delta_indices, lb=0, vtype=GRB.CONTINUOUS)
tot_unfulfilled_demand = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
tot_leftover_supply = m.addVar(lb=0, vtype=GRB.CONTINUOUS)

# create cost variable for delivery
cost = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='total_cost')
#total fixed cost variable
tot_fixed_cost = m.addVar(lb=0, vtype=GRB.CONTINUOUS)

# create additional variables to handle regions
region_indices = [(r, k) for r in regions for k in products]
# regional stock out = 1 if region has stock out
region_stock_out = m.addVars(region_indices, vtype=GRB.BINARY)
# regional flow is sum of flow per product within region
regional_flow = m.addVars(region_indices, lb=0, vtype=GRB.CONTINUOUS)
# nonregional flow is sum of flow per product that originates out of region
nonregional_flow = m.addVars(region_indices, lb=0, vtype=GRB.CONTINUOUS)

## OBJECTIVE FUNCTION
#heavily penalize delta total so that orders are as satisfied when possible
m.setObjective(cost + tot_fixed_cost + 10000 * tot_unfulfilled_demand, sense=GRB.MINIMIZE)

## CONSTRAINTS
# Define cost variable
m.addConstr(cost == sum(flow[t]*costs[t[0]][t[1]]*product_weights[t[2]] for t in indices))

# Define total fixed cost variable
m.addConstr(tot_fixed_cost == sum(fixed_cost_ind[(w,k)]*fixed_costs[w][k] for w in warehouses for k in orders))

#activate fixed costs variable if warehouse is part of order
for w in warehouses:
    for o in orders:
        m.addConstr(fixed_cost_ind[(w,o)] >= sum(flow[(w, o, k)] for k in products if k in orders_data[o])/max_stock)

# meet demand
for o in orders:
    for k in orders_data[o]:
        m.addConstr(sum(flow[(w, o, k)] for w in warehouses) + order_delta[(o, k)] == orders_data[o][k])

# don't exceed stock
for w in warehouses:
    for k in products:
        m.addConstr(sum(flow[(w, o, k)] for o in orders if k in orders_data[o]) + supply_delta[(w,k)]
                            == warehouse_stock[w][k])

# add regional constraints
for r in regions:
    for k in products:
        # define regional flow for product k = sum of product k sent within a region
        m.addConstr(regional_flow[(r,k)] == sum(flow[(w, o, k)] for w in warehouses if warehouse_regions[w]==r
                                                for o in orders if k in orders_data[o] and order_regions[o]==r))
        # define nonregional flow for product k = sum of product k sent to region r from warehouse outside of region r
        m.addConstr(nonregional_flow[(r, k)] == sum(flow[(w, o, k)] for w in warehouses if warehouse_regions[w] != r
                                                    for o in orders if k in orders_data[o] and order_regions[o] == r))
        # make stock out binary active when stock out
        m.addConstr(region_stock[r][k] - regional_flow[(r,k)] >= 1 - region_stock_out[(r,k)])
        # make stock out binary inactive when not stock out (bigM = region_stock[r][k]
        # m.addConstr(region_stock[r][k]*(1-region_stock_out[(r, k)]) >= region_stock[r][k] - regional_flow[(r,k)])
        m.addConstr(region_stock[r][k]*region_stock_out[(r, k)] <= regional_flow[(r,k)])
        # only allow nonregional flow for region/product when stock out is active (bigM = regional_demand[r][k])
        m.addConstr(regional_demand[r][k]*region_stock_out[(r,k)] >= nonregional_flow[(r,k)])


# define order delta total
m.addConstr(tot_unfulfilled_demand == sum(order_delta[t] for t in order_delta_indices))
m.addConstr(tot_leftover_supply == sum(supply_delta[t] for t in supply_delta_indices))

# Solve model
m.optimize()
status_code =   {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
status = m.status
print(f'The optimization status is {status_code[status]}')
print('-------')
# print solution if optimal
if status == 2:
    print(f'Obj function value: {m.objVal}')
    print(f'Total order cost: {cost.X + tot_fixed_cost.X}')
    print(f'Total fixed cost: {tot_fixed_cost.X}')
    print(f'Total unit costs: {cost.X}')
    print('-------')
    print('Region Summary Table')
    print('Region,Product,Regional_stock,Region_stock_out,Regional_Flow,Nonregional_Flow')
    for t in region_indices:
        print(
            f'{t[0]},{t[1]},{region_stock[t[0]][t[1]]},{region_stock_out[t].X},{regional_flow[t].X},{nonregional_flow[t].X}')
    print('-------')
    print('Warehouse\t Warehouse_Region\t Order\t Order_Region\t Product\t Quantity\t Unit_Costs\t')
    for t in indices:
        if flow[t].X != 0:
            print(f'{t[0]},{warehouse_regions[t[0]]},{t[1]},{order_regions[t[1]]},'
                  f'{t[2]},{flow[t].X},{flow[t].X*costs[t[0]][t[1]]*product_weights[t[2]]}')
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


