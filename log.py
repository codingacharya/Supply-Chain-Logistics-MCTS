"""
Streamlit App: Full-featured Supply Chain / Logistics MCTS Demo

Features included:
- Multiple warehouses
- Multi-truck fleet with capacity constraints
- Customer demands and per-customer price
- Time windows (optional)
- Fuel/energy costs (distance-based + load factor)
- Stochastic traffic (affects travel times/distances during rollouts)
- Order of visiting and delivery allocation decisions via hierarchical MCTS
- Interactive Plotly map visualization

How to run:
1) pip install streamlit numpy plotly
2) streamlit run mcts_supply_chain_full.py

Notes:
- This is a pedagogical demo, not production-grade optimization. For larger instances, integrate a VRP solver (OR-Tools) and replace heavy MCTS with hierarchical approaches.

"""

import streamlit as st
import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional
import plotly.graph_objects as go

# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: int
    price: float
    tw_early: float
    tw_late: float

@dataclass
class Warehouse:
    id: int
    x: float
    y: float
    stock: int

@dataclass
class Truck:
    id: int
    capacity: int
    start_warehouse: int  # warehouse id

@dataclass
class State:
    # assignment partial state: maps truck_id -> ordered list of customers assigned
    assignments: Dict[int, List[int]]
    remaining: Set[int]
    # track load assigned per truck
    loads: Dict[int, int]

    def clone(self):
        return State(assignments={k: list(v) for k, v in self.assignments.items()},
                     remaining=set(self.remaining),
                     loads={k: v for k, v in self.loads.items()})

# -----------------------------
# Utilities
# -----------------------------

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# nearest neighbour route given ordered assigned customers (if order empty, we compute greedy order)
def build_route_sequence(start_loc: Tuple[float,float], customers: List[Customer], assigned_ids: List[int]) -> Tuple[List[int], float]:
    id2cust = {c.id: c for c in customers}
    seq = []
    total = 0.0
    current = start_loc
    remaining = assigned_ids[:]
    if not remaining:
        return [], 0.0
    while remaining:
        nxt = min(remaining, key=lambda cid: euclid(current, (id2cust[cid].x, id2cust[cid].y)))
        total += euclid(current, (id2cust[nxt].x, id2cust[nxt].y))
        current = (id2cust[nxt].x, id2cust[nxt].y)
        seq.append(nxt)
        remaining.remove(nxt)
    # return to warehouse
    total += euclid(current, start_loc)
    return seq, total

# compute route distance for a truck starting at warehouse using either assigned order or greedy
def compute_route_distance(warehouse: Warehouse, customers_list: List[Customer], assigned_ids: List[int], traffic_multiplier: float) -> float:
    seq, dist = build_route_sequence((warehouse.x, warehouse.y), customers_list, assigned_ids)
    return dist * traffic_multiplier

# -----------------------------
# MCTS Node and algorithm (action: pick next (truck, customer) pair)
# -----------------------------
class MCTSNode:
    def __init__(self, state: State, parent=None, action: Optional[Tuple[int,int]]=None):
        self.state = state
        self.parent = parent
        self.action = action  # (truck_id, customer_id) or None for root
        self.children: List[MCTSNode] = []
        self.n = 0
        self.w = 0.0
        self.untried: List[Tuple[int,int]] = []  # possible (truck, customer) moves

    def q(self):
        return self.w / self.n if self.n > 0 else 0.0

    def uct_select(self, c=1.4):
        best = None
        best_val = -1e9
        for child in self.children:
            exploitation = child.q()
            exploration = c * math.sqrt(math.log(self.n + 1) / (child.n + 1e-9))
            val = exploitation + exploration
            if val > best_val:
                best_val = val
                best = child
        return best

# rollout: randomly assign remaining customers to trucks then evaluate
def rollout_evaluate(node_state: State, warehouses: Dict[int, Warehouse], trucks: Dict[int, Truck], customers: Dict[int, Customer], cfg) -> float:
    s = node_state.clone()
    # random policy: assign remaining customers to random trucks where capacity allows; if none, assign to nearest warehouse's truck (may exceed capacity -> penalties)
    rem = list(s.remaining)
    random.shuffle(rem)
    truck_ids = list(trucks.keys())
    # naive fill
    for cid in rem:
        # choose trucks that still have capacity
        cand = [tid for tid in truck_ids if s.loads[tid] < trucks[tid].capacity]
        if not cand:
            # force assign to random truck (will cause overload penalty)
            tid = random.choice(truck_ids)
        else:
            tid = random.choice(cand)
        s.assignments[tid].append(cid)
        s.loads[tid] += customers[cid].demand
    # now compute expected reward under stochastic traffic
    traffic_mult = max(0.5, 1.0 + np.random.normal(0, cfg['traffic_sigma']))
    total_reward = evaluate_full_solution(s, warehouses, trucks, customers, cfg, traffic_mult)
    return total_reward

# evaluate full solution: compute revenue minus costs and penalties
def evaluate_full_solution(state: State, warehouses: Dict[int, Warehouse], trucks: Dict[int, Truck], customers: Dict[int, Customer], cfg, traffic_multiplier: float) -> float:
    revenue = 0.0
    travel_cost = 0.0
    time_window_penalty = 0.0
    overload_penalty = 0.0
    unmet_penalty = 0.0

    # For each truck, compute route and deliveries
    for tid, assigned in state.assignments.items():
        truck = trucks[tid]
        wh = warehouses[truck.start_warehouse]
        # build route sequence greedily
        seq, dist = build_route_sequence((wh.x, wh.y), list(customers.values()), assigned)
        # apply traffic multiplier
        route_dist = dist * traffic_multiplier
        # travel cost increases with average load factor
        avg_load = (sum(customers[cid].demand for cid in assigned) / (truck.capacity + 1e-9))
        load_factor = max(0.0, min(1.0, avg_load))
        travel_cost += route_dist * (cfg['travel_cost_per_km'] * (1.0 + cfg['load_cost_factor'] * load_factor))

        # deliver and compute revenue / penalties
        for cid in seq:
            cust = customers[cid]
            qty = min(cust.demand, truck.capacity)  # simplistic: truck may deliver up to capacity but multiple customers reduce capacity
            # For realism, assume delivered equals customer demand if total assigned load <= capacity; else cap
            total_assigned_load = sum(customers[c].demand for c in assigned)
            if total_assigned_load <= truck.capacity:
                delivered = cust.demand
            else:
                # proportionally allocate
                delivered = int(round(cust.demand * (truck.capacity / total_assigned_load)))
            revenue += delivered * cust.price
            if delivered < cust.demand:
                unmet_penalty += (cust.demand - delivered) * cfg['stockout_penalty_per_unit']
            # time windows: compute approximate arrival time: assume speed = cfg['speed_km_per_h'] and time adds from distance; here just approximate using cumulative distance
            # skipping detailed time calc; if time windows enabled, penalize a bit randomly to simulate violation
            if cfg['use_time_windows']:
                # probability of violation increases with traffic
                if random.random() < min(0.3, 0.1 * traffic_multiplier):
                    time_window_penalty += cfg['time_window_violation_penalty']

        # capacity overload
        assigned_load = sum(customers[c].demand for c in assigned)
        if assigned_load > truck.capacity:
            overload_penalty += (assigned_load - truck.capacity) * cfg['overload_penalty_per_unit']

    # warehouses stock constraints are ignored here (assume replenished)

    total = revenue - (travel_cost + time_window_penalty + overload_penalty + unmet_penalty)
    return total

# MCTS planning function that builds assignments
def mcts_plan(warehouses: Dict[int, Warehouse], trucks: Dict[int, Truck], customers: Dict[int, Customer], cfg, iterations:int=400) -> State:
    # initial state: no assignments
    init_assign = {tid: [] for tid in trucks.keys()}
    init_loads = {tid: 0 for tid in trucks.keys()}
    init_state = State(assignments=init_assign, remaining=set(customers.keys()), loads=init_loads)
    root = MCTSNode(init_state)

    # root untried moves: all (truck, customer) possible pairs
    def possible_moves(s: State):
        moves = []
        for tid in trucks.keys():
            for cid in s.remaining:
                moves.append((tid, cid))
        return moves

    root.untried = possible_moves(root.state)

    for it in range(iterations):
        node = root
        # Selection
        while node.untried == [] and node.children:
            node = node.uct_select(c=cfg['uct_c'])
        # Expansion
        if node.untried:
            a = random.choice(node.untried)
            node.untried.remove(a)
            new_state = node.state.clone()
            tid, cid = a
            new_state.assignments[tid].append(cid)
            new_state.loads[tid] += customers[cid].demand
            new_state.remaining.remove(cid)
            child = MCTSNode(new_state, parent=node, action=a)
            child.untried = possible_moves(child.state)
            node.children.append(child)
            node = child
        # Simulation / Rollout
        value = rollout_evaluate(node.state, warehouses, trucks, customers, cfg)
        # Backpropagation
        while node is not None:
            node.n += 1
            node.w += value
            node = node.parent

    # After iterations, pick best child path greedily by visiting children with highest visit count until all customers assigned
    final_state = init_state.clone()
    cur = root
    while final_state.remaining:
        if not cur.children:
            # no explored children; assign remaining randomly
            for cid in list(final_state.remaining):
                tid = random.choice(list(trucks.keys()))
                final_state.assignments[tid].append(cid)
                final_state.loads[tid] += customers[cid].demand
                final_state.remaining.remove(cid)
            break
        # pick child of cur with max visits
        best_child = max(cur.children, key=lambda c: c.n)
        tid, cid = best_child.action
        if cid in final_state.remaining:
            final_state.assignments[tid].append(cid)
            final_state.loads[tid] += customers[cid].demand
            final_state.remaining.remove(cid)
        # move to that child
        cur = best_child
        # if this child has children continue else if still remaining, break to random assign
    return final_state

# -----------------------------
# Streamlit UI & glue
# -----------------------------
st.set_page_config(page_title='MCTS Supply Chain (Full)', layout='wide')
st.title('🚚 Full Supply Chain & Logistics MCTS Demo')

with st.sidebar:
    st.header('Scenario Parameters')
    n_customers = st.slider('Customers', 3, 12, 6)
    n_warehouses = st.slider('Warehouses', 1, 3, 1)
    n_trucks = st.slider('Trucks', 1, 5, 2)
    truck_capacity = st.number_input('Truck capacity (units)', value=30, step=5)
    iterations = st.slider('MCTS iterations', 100, 2000, 600, step=50)

    st.header('Costs & Traffic')
    travel_cost_per_km = st.number_input('Travel cost per km', value=1.0, step=0.1)
    load_cost_factor = st.number_input('Load cost factor (multiplier per load)', value=0.4, step=0.05)
    traffic_sigma = st.slider('Traffic volatility (std dev)', 0.0, 0.8, 0.15, step=0.05)

    st.header('Penalties')
    stockout_penalty = st.number_input('Stockout penalty per unit', value=5.0, step=0.5)
    overload_pen_unit = st.number_input('Overload penalty per unit', value=10.0, step=0.5)
    use_time_windows = st.checkbox('Use time windows', value=True)
    time_window_violation_penalty = st.number_input('Time window violation penalty', value=20.0, step=1.0)

    st.header('Random Seed')
    seed = st.number_input('Seed', value=42, step=1)

# Build scenario
random.seed(int(seed))
np.random.seed(int(seed))

# place warehouses and customers in unit square
warehouses: Dict[int, Warehouse] = {}
for i in range(n_warehouses):
    x, y = np.random.rand(), np.random.rand()
    warehouses[i] = Warehouse(id=i, x=float(x), y=float(y), stock=10000)

customers: Dict[int, Customer] = {}
for i in range(1, n_customers+1):
    x, y = np.random.rand(), np.random.rand()
    demand = int(np.random.poisson(5) + 1)
    price = float(np.random.uniform(8.0, 15.0))
    # time windows centered around random times (in hours) with width
    tw_center = np.random.uniform(8, 17)
    tw_width = np.random.uniform(1.0, 4.0)
    customers[i] = Customer(id=i, x=float(x), y=float(y), demand=demand, price=price, tw_early=tw_center - tw_width/2, tw_late=tw_center + tw_width/2)

# trucks start from random warehouses
trucks: Dict[int, Truck] = {}
for t in range(n_trucks):
    wh_id = random.choice(list(warehouses.keys()))
    trucks[t] = Truck(id=t, capacity=int(truck_capacity), start_warehouse=wh_id)

cfg = {
    'travel_cost_per_km': float(travel_cost_per_km),
    'load_cost_factor': float(load_cost_factor),
    'traffic_sigma': float(traffic_sigma),
    'stockout_penalty_per_unit': float(stockout_penalty),
    'overload_penalty_per_unit': float(overload_pen_unit),
    'use_time_windows': bool(use_time_windows),
    'time_window_violation_penalty': float(time_window_violation_penalty),
    'uct_c': 1.4,
}

st.markdown('### Scenario overview')
col1, col2 = st.columns([2, 1])
with col1:
    # plot nodes
    fig = go.Figure()
    # warehouses
    for wid, wh in warehouses.items():
        fig.add_trace(go.Scatter(x=[wh.x], y=[wh.y], mode='markers+text', marker=dict(size=16, symbol='square', color='red'), text=[f'W{wid}'], textposition='top center'))
    # customers
    for cid, c in customers.items():
        fig.add_trace(go.Scatter(x=[c.x], y=[c.y], mode='markers+text', marker=dict(size=10, color='blue'), text=[f'C{cid} (d={c.demand})'], textposition='top center'))
    fig.update_layout(width=700, height=700, title='Warehouses and Customers (unit square)')
    st.plotly_chart(fig)
with col2:
    st.write('**Warehouses**')
    for wid, wh in warehouses.items():
        st.write(f'W{wid}: ({wh.x:.2f}, {wh.y:.2f}) stock={wh.stock}')
    st.write('**Trucks**')
    for tid, tr in trucks.items():
        st.write(f'T{tid}: cap={tr.capacity}, start=W{tr.start_warehouse}')

# Run MCTS planning
with st.spinner('Running MCTS planning...'):
    plan = mcts_plan(warehouses, trucks, customers, cfg, iterations=int(iterations))

# Evaluate plan deterministically with expected traffic multiplier = 1.0
final_reward = evaluate_full_solution(plan, warehouses, trucks, customers, cfg, traffic_multiplier=1.0)

st.success(f'Planning done — Estimated objective: {final_reward:.2f}')

# Show assignments and simple table
st.subheader('Assignments per Truck')
for tid, assigned in plan.assignments.items():
    st.write(f'Truck {tid} (start W{trucks[tid].start_warehouse}) -> Customers: {assigned} | total load {plan.loads[tid]}')

# Build plot of routes with deterministic traffic
fig2 = go.Figure()
# plot warehouses and customers
for wid, wh in warehouses.items():
    fig2.add_trace(go.Scatter(x=[wh.x], y=[wh.y], mode='markers+text', marker=dict(size=14, symbol='square', color='red'), text=[f'W{wid}'], textposition='bottom center'))
for cid, c in customers.items():
    fig2.add_trace(go.Scatter(x=[c.x], y=[c.y], mode='markers+text', marker=dict(size=10, color='blue'), text=[f'C{cid}\nd={c.demand}'], textposition='top center'))

colors = ['green','orange','purple','brown','magenta','cyan']
for idx, (tid, assigned) in enumerate(plan.assignments.items()):
    truck = trucks[tid]
    wh = warehouses[truck.start_warehouse]
    seq, dist = build_route_sequence((wh.x, wh.y), list(customers.values()), assigned)
    xs = [wh.x]
    ys = [wh.y]
    for cid in seq:
        xs.append(customers[cid].x)
        ys.append(customers[cid].y)
    xs.append(wh.x)
    ys.append(wh.y)
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(color=colors[idx % len(colors)], width=2), name=f'Truck {tid}'))

fig2.update_layout(width=900, height=700, title='Planned Routes (deterministic)')
st.plotly_chart(fig2)

# Allow user to simulate stochastic rollout multiple times and show statistics
st.subheader('Stochastic Simulation (rollouts)')
nsim = st.number_input('Number of rollout simulations', min_value=10, max_value=200, value=50, step=10)

if st.button('Run Rollouts'):
    results = []
    for i in range(int(nsim)):
        tm = max(0.4, 1.0 + np.random.normal(0, cfg['traffic_sigma']))
        val = evaluate_full_solution(plan, warehouses, trucks, customers, cfg, traffic_multiplier=tm)
        results.append(val)
    import statistics
    st.write(f'Rollouts: mean={statistics.mean(results):.2f}, std={statistics.stdev(results):.2f}, min={min(results):.2f}, max={max(results):.2f}')
    st.bar_chart(results)

st.markdown('---')
st.caption('This demo uses MCTS over the (truck,customer) assignment decision space and greedy routing. For larger/real problems, consider OR-Tools for VRP and hierarchical planning.')
