
import pandas as pd
from pulp import *

# === CARGA DE DATOS ===
archivos = [
    "Suppliers", "Plants", "Receptions", "Bundles", "Simple_products",
    "Simple_per_bundle", "Distances", "Surcharge", "Orders",
    "Multimodal_transport", "Route_supplier"
]
df = {name: pd.read_csv(f"{name}.csv", sep=";") for name in archivos}
for name in df:
    df[name].columns = df[name].columns.str.strip()

# === CONJUNTOS ===
P = df["Suppliers"]['SupplierID'].unique()
F = df["Plants"]['PlantID'].unique()
I = df["Receptions"]['ReceptionID'].unique()
B = df["Bundles"]['BundleID'].unique()
M = df["Simple_products"]['SimpleProductID'].unique()

# === VARIABLES ===
model = LpProblem("Modelo_Logistico_PDF", LpMinimize)
Q = LpVariable.dicts("Q", ((m, p, i) for m in M for p in P for i in I), lowBound=0)
QB = LpVariable.dicts("QB", ((b, p, i) for b in B for p in P for i in I), lowBound=0)
X = LpVariable.dicts("X", ((m, i, f) for m in M for i in I for f in F), lowBound=0)
XB = LpVariable.dicts("XB", ((b, i, f) for b in B for i in I for f in F), lowBound=0)
Z = LpVariable.dicts("Z", ((i, f) for i in I for f in F), cat="Binary")

# === PARÁMETROS ===
# Costos de compra
c = {(m, p): df["Suppliers"].query("ProductProduced == @m and SupplierID == @p")["PerUnitProductCost"].values[0]
     for m in M for p in P if not df["Suppliers"].query("ProductProduced == @m and SupplierID == @p").empty}

# Costos de transporte proveedor -> intake
r = {(p, i): df["Route_supplier"].query("SupplierID == @p and ReceptionID == @i")["TransportationCostPerUnit"].values[0]
     for p in P for i in I if not df["Route_supplier"].query("SupplierID == @p and ReceptionID == @i").empty}

# Costos transporte intake -> planta
t = {(i, f): df["Multimodal_transport"].query("ReceptionID == @i and PlantID == @f")["TransportationCostPerUnitAutomatic"].values[0]
     for i in I for f in F if not df["Multimodal_transport"].query("ReceptionID == @i and PlantID == @f").empty}

# Costos handling planta
h = dict(zip(df["Plants"]["PlantID"], df["Plants"]["PlantHandlingCostPerUnitAutomatic"]))

# Unidades en bundle
u = {(b, m): row["UnitsInBundle"] for _, row in df["Simple_per_bundle"].iterrows()}

# Demanda
d = {(f, m): df["Orders"].query("PlantID == @f and SimpleProductID == @m")["NumSimpleProductsOrdered"].sum()
     for f in F for m in M}

# Capacidad proveedor
a = {(p, m): df["Suppliers"].query("SupplierID == @p and ProductProduced == @m")["ProductAvailability"].sum()
     for p in P for m in M}

# === FUNCIÓN OBJETIVO ===
model += (
    lpSum(Q[m, p, i] * (c[m, p] + r.get((p, i), 0)) for m in M for p in P for i in I if (m, p) in c) +
    lpSum(QB[b, p, i] * (
        sum(u.get((b, m), 0) * c.get((m, p), 0) for m in M) + r.get((p, i), 0))
        for b in B for p in P for i in I) +
    lpSum(X[m, i, f] * (t.get((i, f), 0) + h.get(f, 0)) for m in M for i in I for f in F) +
    lpSum(XB[b, i, f] * (t.get((i, f), 0) + h.get(f, 0)) for b in B for i in I for f in F)
), "Costo_Total"

# === RESTRICCIONES ===
# Satisfacción demanda
for f in F:
    for m in M:
        model += (
            lpSum(X[m, i, f] for i in I) +
            lpSum(XB[b, i, f] * u.get((b, m), 0)
                  for b in B for i in I if (b, m) in u)
        ) >= d.get((f, m), 0), f"Demanda_{f}_{m}"

# Capacidad proveedor
for p in P:
    for m in M:
        model += (
            lpSum(Q[m, p, i] for i in I) +
            lpSum(QB[b, p, i] * u.get((b, m), 0)
                  for b in B for i in I if (b, m) in u)
        ) <= a.get((p, m), 0), f"Capacidad_{p}_{m}"

# Balance intake
for m in M:
    for i in I:
        model += lpSum(X[m, i, f] for f in F) <= lpSum(Q[m, p, i] for p in P), f"Balance_MP_{m}_{i}"

for b in B:
    for i in I:
        model += lpSum(XB[b, i, f] for f in F) <= lpSum(QB[b, p, i] for p in P), f"Balance_Bundle_{b}_{i}"

# Conectividad intake-planta
for f in F:
    model += lpSum(Z[i, f] for i in I) >= 2, f"Min_Intake_{f}"
    model += lpSum(Z[i, f] for i in I) <= 5, f"Max_Intake_{f}"

for i in I:
    for f in F:
        model += (
            lpSum(X[m, i, f] for m in M) +
            lpSum(XB[b, i, f] for b in B)
        ) >= 0.001 * Z[i, f], f"ActivaZ_{i}_{f}"

# === RESOLVER ===
model.solve()
print("Estado:", LpStatus[model.status])
print("Costo total:", value(model.objective))
for v in model.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)
