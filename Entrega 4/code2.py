
import pandas as pd
from pulp import *

# Cargar datos
archivos = [
    "Suppliers", "Plants", "Receptions", "Bundles", "Simple_products",
    "Simple_per_bundle", "Distances", "Surcharge", "Orders",
    "Multimodal_transport", "Route_supplier"
]
df = {name: pd.read_csv(f"{name}.csv", sep=";") for name in archivos}

# Corregir espacios en los nombres de columnas
for name in df:
    df[name].columns = df[name].columns.str.strip()

# Conjuntos
P = df["Suppliers"]['SupplierID'].unique()
M = df["Simple_products"]['SimpleProductID'].unique()
B = df["Bundles"]['BundleID'].unique()
I = df["Receptions"]['ReceptionID'].unique()
F = df["Plants"]['PlantID'].unique()

# Modelo
model = LpProblem("Modelo_Logistico_Alineado_g1", LpMinimize)

Q = LpVariable.dicts("Q", ((m, p, i) for m in M for p in P for i in I), lowBound=0)
QB = LpVariable.dicts("QB", ((b, p, i) for b in B for p in P for i in I), lowBound=0)
X = LpVariable.dicts("X", ((m, i, f) for m in M for i in I for f in F), lowBound=0)
XB = LpVariable.dicts("XB", ((b, i, f) for b in B for i in I for f in F), lowBound=0)
Z = LpVariable.dicts("Z", ((i, f) for i in I for f in F), cat="Binary")

# Costos auxiliares
bundle_cost = {}
for b in B:
    simples = df["Simple_per_bundle"].query("BundleID == @b")
    costo = 0
    for _, row in simples.iterrows():
        mp = row['SimpleProductID']
        unidades = row['UnitsInBundle']
        costop = df["Suppliers"].query("ProductProduced == @mp")['PerUnitProductCost'].mean()
        costo += unidades * costop
    bundle_cost[b] = costo

transport_cost = {}
for _, row in df["Multimodal_transport"].iterrows():
    i, f = row['ReceptionID'], row['PlantID']
    transport_cost[(i, f)] = row['TransportationCostPerUnitAutomatic']

handling_cost_plant = dict(zip(df["Plants"]['PlantID'], df["Plants"]['PlantHandlingCostPerUnitAutomatic']))

route_cost = {}
for _, row in df["Route_supplier"].iterrows():
    route_cost[(row['SupplierID'], row['ReceptionID'])] = row['TransportationCostPerUnit']

# Función objetivo
model += (
    lpSum(Q[m, p, i] * (
        df["Suppliers"].query("SupplierID == @p and ProductProduced == @m")['PerUnitProductCost'].values[0] +
        route_cost.get((p, i), 0))
        for m in M for p in P for i in I
        if not df["Suppliers"].query("SupplierID == @p and ProductProduced == @m").empty
    ) +
    lpSum(QB[b, p, i] * (bundle_cost[b] + route_cost.get((p, i), 0))
        for b in B for p in P for i in I) +
    lpSum(X[m, i, f] * (transport_cost.get((i, f), 0) + handling_cost_plant.get(f, 0))
        for m in M for i in I for f in F) +
    lpSum(XB[b, i, f] * (transport_cost.get((i, f), 0) + handling_cost_plant.get(f, 0))
        for b in B for i in I for f in F)
), "Costo_Total"

# Restricción demanda
for f in F:
    for m in M:
        demanda = df["Orders"].query("PlantID == @f and SimpleProductID == @m")['NumSimpleProductsOrdered'].sum()
        model += (
            lpSum(X[m, i, f] for i in I) +
            lpSum(
                XB[b, i, f] * df["Simple_per_bundle"].query("BundleID == @b and SimpleProductID == @m")['UnitsInBundle'].values[0]
                for b in B if not df["Simple_per_bundle"].query("BundleID == @b and SimpleProductID == @m").empty
                for i in I
            )
        ) >= demanda, f"Demanda_{f}_{m}"

# Restricción capacidad proveedor
for p in P:
    for m in M:
        cap = df["Suppliers"].query("SupplierID == @p and ProductProduced == @m")['ProductAvailability'].sum()
        model += (
            lpSum(Q[m, p, i] for i in I) +
            lpSum(
                QB[b, p, i] * df["Simple_per_bundle"].query("BundleID == @b and SimpleProductID == @m")['UnitsInBundle'].sum()
                for b in B for i in I
                if not df["Simple_per_bundle"].query("BundleID == @b and SimpleProductID == @m").empty
            )
        ) <= cap, f"Capacidad_{p}_{m}"

# Balance intake
for m in M:
    for i in I:
        model += lpSum(X[m, i, f] for f in F) <= lpSum(Q[m, p, i] for p in P), f"BalanceMP_{m}_{i}"

for b in B:
    for i in I:
        model += lpSum(XB[b, i, f] for f in F) <= lpSum(QB[b, p, i] for p in P), f"BalanceBundle_{b}_{i}"

# Conectividad intake-planta
for f in F:
    model += lpSum(Z[i, f] for i in I) >= 2, f"MinConex_{f}"
    model += lpSum(Z[i, f] for i in I) <= 5, f"MaxConex_{f}"

for i in I:
    for f in F:
        model += (
            lpSum(X[m, i, f] for m in M) +
            lpSum(XB[b, i, f] for b in B)
        ) >= 0.001 * Z[i, f], f"ActivaZ_{i}_{f}"

# Resolver
model.solve()
print("Estado:", LpStatus[model.status])
print("Costo total:", value(model.objective))
for v in model.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)
