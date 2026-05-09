import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpStatus, LpVariable, LpMaximize, lpSum, value

plt.style.use('seaborn-v0_8')

products = ['Tables', 'Chairs']
profit = {'Tables': 70, 'Chairs': 40}
labor = {'Tables': 5, 'Chairs': 2}
wood = {'Tables': 8, 'Chairs': 4}
max_demand = {'Tables': 15, 'Chairs': 35}

if __name__ == '__main__':
    data = pd.DataFrame({
        'Profit': [profit[p] for p in products],
        'Labor (hours)': [labor[p] for p in products],
        'Wood (board-feet)': [wood[p] for p in products],
        'Max Demand': [max_demand[p] for p in products]
    }, index=products)

    print('Problem data:')
    print(data)
    print()

    problem = LpProblem('Furniture_Production', LpMaximize)
    x = {p: LpVariable(p.lower(), lowBound=0, cat='Continuous') for p in products}

    problem += lpSum(profit[p] * x[p] for p in products), 'Total_Profit'
    problem += lpSum(labor[p] * x[p] for p in products) <= 80, 'Labor_Hours'
    problem += lpSum(wood[p] * x[p] for p in products) <= 120, 'Wood_Consumption'
    for p in products:
        problem += x[p] <= max_demand[p], f'Max_{p}'

    problem.solve()
    status = LpStatus[problem.status]
    optimal_values = {p: value(x[p]) for p in products}
    optimal_profit = value(problem.objective)

    results = pd.DataFrame({
        'Product': products,
        'Optimal Quantity': [optimal_values[p] for p in products],
        'Profit per Unit': [profit[p] for p in products]
    })

    print(results)
    print(f'Optimal profit: ${optimal_profit:,.2f}')
    print(f'Status: {status}')

    scenario_labor = [60, 80, 100]
    scenario_rows = []
    for labor_capacity in scenario_labor:
        scenario = LpProblem(f'Furniture_Production_Labor_{labor_capacity}', LpMaximize)
        y = {p: LpVariable(f"{p.lower()}_{labor_capacity}", lowBound=0, cat='Continuous') for p in products}
        scenario += lpSum(profit[p] * y[p] for p in products)
        scenario += lpSum(labor[p] * y[p] for p in products) <= labor_capacity
        scenario += lpSum(wood[p] * y[p] for p in products) <= 120
        for p in products:
            scenario += y[p] <= max_demand[p]
        scenario.solve()
        scenario_rows.append({
            'Labor capacity': labor_capacity,
            'Tables': value(y['Tables']),
            'Chairs': value(y['Chairs']),
            'Profit': value(scenario.objective)
        })

    scenario_df = pd.DataFrame(scenario_rows)
    print('\nSensitivity analysis: labor capacity')
    print(scenario_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results['Product'], results['Optimal Quantity'], color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Optimal Weekly Production Plan')
    ax.set_ylabel('Quantity')
    for index, value_qty in enumerate(results['Optimal Quantity']):
        ax.text(index, value_qty + 0.5, int(value_qty), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
