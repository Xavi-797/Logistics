import pandas as pd
from pulp import *

# === FUNCIONES ===
def read_clean_csv(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

# === CARGA DE DATOS ===
folder = "cases"

suppliers = read_clean_csv(f"{folder}/Suppliers.csv")
plants = read_clean_csv(f"{folder}/Plants.csv")
receptions = read_clean_csv(f"{folder}/Receptions.csv")
bundles = read_clean_csv(f"{folder}/Bundles.csv")
simple_products = read_clean_csv(f"{folder}/Simple_products.csv")
simple_per_bundle = read_clean_csv(f"{folder}/Simple_per_bundle.csv")
orders = read_clean_csv(f"{folder}/Orders.csv")
multimodal = read_clean_csv(f"{folder}/Multimodal_transport.csv")
route_supplier = read_clean_csv(f"{folder}/Route_supplier.csv")
surcharge = read_clean_csv(f"{folder}/Surcharge.csv")

# === CONJUNTOS ===
P = suppliers["supplier_id"].unique()
F = plants["plant_id"].unique()
I = receptions["reception_id"].unique()
M = simple_products["product_id"].unique()
B = bundles["bundle_id"].unique()
T = ["InHouse", "TR1", "TR2"]

# === PARÁMETROS ===
unit_cost = {(m, p): cost for _, (p, m, cost) in suppliers[['supplier_id', 'product_id', 'unit_cost']].iterrows()}
capacity_supplier = {(m, p): cap for _, (p, m, cap) in suppliers[['supplier_id', 'product_id', 'availability']].iterrows()}
unit_cost_bundles = {(b, p): cost for _, (b, p, cost) in bundles[['bundle_id', 'supplier_id', 'unit_cost']].iterrows()}
transport_cost = {(p, i): cost for _, (p, i, cost) in route_supplier[['SupplierID', 'ReceptionID', 'TransportationCostPerUnit']].iterrows()}
mode_capacity = {t: cost_lookup(multimodal, {"mode": t}, "capacity") for t in T}

# === COMPATIBILIDAD ===
compatible_modes_m = {
    m: set(multimodal[multimodal["product_id"] == m]["mode"]) for m in M
}

compatible_modes_b = {
    b: set(multimodal[multimodal["bundle_id"] == b]["mode"]) for b in B
}

# === MODELO ===
model = LpProblem("Logistics_Planning_Model", LpMinimize)

# === VARIABLES ===
Q = LpVariable.dicts("Q", ((m, p, i) for m in M for p in P for i in I), lowBound=0, cat='Continuous')
QB = LpVariable.dicts("QB", ((b, p, i) for b in B for p in P for i in I), lowBound=0, cat='Continuous')
X = LpVariable.dicts("X", ((m, i, f, t) for m in M for i in I for f in F for t in T), lowBound=0, cat='Continuous')
XB = LpVariable.dicts("XB", ((b, i, f, t) for b in B for i in I for f in F for t in T), lowBound=0, cat='Continuous')
Z = LpVariable.dicts("Z", ((i, f) for i in I for f in F), cat='Binary')
YB = LpVariable.dicts("YB", ((b, i, f) for b in B for i in I for f in F), cat='Binary')

M_big = 1e6

def cost_lookup(df, conds, col):
    row = df
    for k, v in conds.items():
        row = row[row[k] == v]
    if not row.empty:
        return row.iloc[0][col]
    else:
        return 0

# === FUNCIÓN OBJETIVO ===
model += (
    lpSum((unit_cost.get((m, p), 0) + transport_cost.get((p, i), 0)) * Q[(m, p, i)]
          for m in M for p in P for i in I) +
    lpSum((unit_cost_bundles.get((b, p), 0) + transport_cost.get((p, i), 0)) * QB[(b, p, i)]
          for b in B for p in P for i in I) +
    lpSum((cost_lookup(multimodal, {"ReceptionID": i, "PlantID": f}, f"cost_{t}") +
           cost_lookup(receptions, {"reception_id": i}, f"cm_intake_{t.lower()}") +
           cost_lookup(plants, {"plant_id": f}, f"cm_plant_{t.lower()}") +
           cost_lookup(surcharge, {"ReceptionID": i, "PlantID": f}, f"tax_{t}")) * X[(m, i, f, t)]
          for m in M for i in I for f in F for t in T) +
    lpSum((cost_lookup(multimodal, {"ReceptionID": i, "PlantID": f}, f"cost_{t}") +
           cost_lookup(receptions, {"reception_id": i}, f"cm_intake_{t.lower()}") +
           cost_lookup(plants, {"plant_id": f}, f"cm_plant_{t.lower()}") +
           cost_lookup(surcharge, {"ReceptionID": i, "PlantID": f}, f"tax_{t}")) * XB[(b, i, f, t)]
          for b in B for i in I for f in F for t in T)
)

# === RESTRICCIONES ===
# Demanda
for f in F:
    for m in M:
        demand = orders[(orders["plant_id"] == f) & (orders["product_id"] == m)]["quantity"].sum()
        supply_from_simple = lpSum(X[(m, i, f, t)] for i in I for t in T)
        supply_from_bundles = lpSum(XB[(b, i, f, t)] * row["quantity"]
            for b in B for i in I for t in T
            for _, row in simple_per_bundle[(simple_per_bundle["bundle_id"] == b) & (simple_per_bundle["product_id"] == m)].iterrows())
        model += supply_from_simple + supply_from_bundles >= demand

# Capacidad del proveedor
for p in P:
    for m in M:
        cap = capacity_supplier.get((m, p), 0)
        sum_simple = lpSum(Q[(m, p, i)] for i in I)
        sum_bundle = lpSum(QB[(b, p, i)] * row["quantity"]
            for b in B for i in I
            for _, row in simple_per_bundle[(simple_per_bundle["bundle_id"] == b) & (simple_per_bundle["product_id"] == m)].iterrows())
        model += sum_simple + sum_bundle <= cap

# Compatibilidad modo-materia y capacidad transporte
for t in T:
    cap = mode_capacity[t]
    for i in I:
        for f in F:
            sum_simple = lpSum(X[(m, i, f, t)] for m in M if t in compatible_modes_m.get(m, []))
            sum_bundles = lpSum(XB[(b, i, f, t)] for b in B if t in compatible_modes_b.get(b, []))
            model += sum_simple + sum_bundles <= cap

# Balance en intake
for i in I:
    for m in M:
        total_in = lpSum(Q[(m, p, i)] for p in P)
        total_out = lpSum(X[(m, i, f, t)] for f in F for t in T)
        model += total_out <= total_in
    for b in B:
        total_in_b = lpSum(QB[(b, p, i)] for p in P)
        total_out_b = lpSum(XB[(b, i, f, t)] for f in F for t in T)
        model += total_out_b <= total_in_b

# Conectividad intake-planta
for f in F:
    model += lpSum(Z[(i, f)] for i in I) >= 2
    model += lpSum(Z[(i, f)] for i in I) <= 5
for i in I:
    for f in F:
        model += lpSum(X[(m, i, f, t)] + XB[(b, i, f, t)] for m in M for b in B for t in T) >= 0.01 * Z[(i, f)]

# Compatibilidad proveedor-producto
for m in M:
    for p in P:
        if (m, p) not in unit_cost:
            for i in I:
                model += Q[(m, p, i)] == 0
for b in B:
    for p in P:
        if (b, p) not in unit_cost_bundles:
            for i in I:
                model += QB[(b, p, i)] == 0

# Unicidad de ruta para cada bundle
for b in B:
    for i in I:
        for f in F:
            model += lpSum(XB[(b, i, f, t)] for t in T) <= M_big * YB[(b, i, f)]
for b in B:
    model += lpSum(YB[(b, i, f)] for i in I for f in F) <= 1

# Resolver
model.solve()

print(f"Status: {LpStatus[model.status]}")
print(f"Costo total optimizado: {value(model.objective):,.2f}")
