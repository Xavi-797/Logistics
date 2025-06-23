import pandas as pd
import os
from pulp import *

# ----------- Cargar datos con manejo de errores -----------
archivos = {
    "bundles": "bundles.csv",
    "distances": "distances.csv",
    "multimodal_transport": "multimodal_transport.csv",
    "orders": "orders.csv",
    "plants": "plants.csv",
    "receptions": "receptions.csv",
    "route_supplier": "route_supplier.csv",
    "simple_per_bundle": "simple_per_bundle.csv",
    "simple_products": "simple_products.csv",
    "suppliers": "suppliers.csv",
    "surcharge": "surcharge.csv"
}

dataframes = {}
for k, v in archivos.items():
    if os.path.exists(v):
        try:
            dataframes[k] = pd.read_csv(v, sep=";")
        except Exception as e:
            print(f"⚠️ Error al leer {v}: {e}")
    else:
        print(f"🔍 Archivo no encontrado, ignorando: {v}")

# ----------- Conjuntos -----------
def extraer_conjuntos(df):
    """Esta funcion extrae los conjuntos de los archivos"""
    P = df["suppliers"]["SupplierID"].dropna().unique().tolist()
    M = df["simple_products"]["SimpleProductID"].dropna().unique().tolist()
    B = df["bundles"]["BundleID"].dropna().unique().tolist()
    I = df["receptions"]["ReceptionID"].dropna().unique().tolist()
    F = df["plants"]["PlantID"].dropna().unique().tolist()
    T = [col.replace("TransportationCostPerUnit", "")
        for col in df["multimodal_transport"].columns 
            if "TransportationCostPerUnit" in col]
    return P, M, B, I, F, T

P, M, B, I, F, T = extraer_conjuntos(dataframes)


# ----------- Parámetros -----------
def construir_parametros(df):
    CAdq = {(r["SupplierID"], r["ProductProduced"]): r["PerUnitProductCost"]
            for _, r in df["suppliers"].iterrows()}
    CapProv = {(r["SupplierID"], r["ProductProduced"]): r["ProductAvailability"]
               for _, r in df["suppliers"].iterrows()}
    CEnv = {(r["SupplierID"], r["ReceptionID"]): r["TransportationCostPerUnit"]
            for _, r in df["route_supplier"].iterrows()}
    Comp_mt = {(r["SimpleProductID"], t): int(r["IsAutomaticCompatible"]) 
        if t == "Automatic" else 1
               for _, r in df["simple_products"].iterrows() for t in T}
    Comp_bt = {(r["BundleID"], t): int(r["IsAutomaticCompatible"]) 
        if t == "Automatic" 
        else 1
            for _, r in df["bundles"].iterrows() for t in T}
    B_prop = {(r["BundleID"], r["SimpleProductID"]): r["UnitsInBundle"]
              for _, r in df["simple_per_bundle"].iterrows()}
    
    orders_df = df["orders"]
    orders_df.columns = [col.strip() for col in orders_df.columns]
    assert "PlantID" in orders_df.columns
    assert "SimpleProductID" in orders_df.columns
    assert "NumSimpleProductsOrdered" in orders_df.columns

    D_raw = {(row["PlantID"], row["SimpleProductID"]): row["NumSimpleProductsOrdered"]
             for _, row in orders_df.iterrows()}

    Reg = {(r["ReceptionID"], r["PlantID"]): int(r["DistanceKilometers"] <= 50)
           for _, r in df["distances"].iterrows()}
    alpha = {(r["ReceptionID"], r["PlantID"]): int(r["DistanceKilometers"] <= 30)
             for _, r in df["distances"].iterrows()}
    CST = {(r["ReceptionID"], r["PlantID"], t): r[f"TransportationCostPerUnit{t}"]
           for _, r in df["multimodal_transport"].iterrows() for t in T}
    Imp = {(r["ReceptionID"], t, r["PlantID"]): r["HighwayTaxPerWeightKilogram"] if r["HasHighway"] else 0
           for _, r in df["multimodal_transport"].iterrows() for t in T}
    CManiIntake = {(r["ReceptionID"], t): r[f"ReceptionHandlingCostPerUnit{t}"]
                   for _, r in df["receptions"].iterrows() for t in ["Automatic", "Manual"]}
    CManiPlanta = {(r["PlantID"], t): r[f"PlantHandlingCostPerUnit{t}"]
                   for _, r in df["plants"].iterrows() for t in ["Automatic", "Manual"]}

    return CAdq, CapProv, CEnv, Comp_mt, Comp_bt, B_prop, D_raw, Reg, alpha, CST, Imp, CManiIntake, CManiPlanta


CAdq, CapProv, CEnv, Comp_mt, Comp_bt, B_prop, D_raw, Reg, alpha, CST, Imp, CManiIntake, CManiPlanta = construir_parametros(dataframes)


# ----------- Filtrar demanda válida -----------
D = {k: v for k, v in D_raw.items() if k[0] in F and k[1] in M}

print("P:", len(P), "M:", len(M), "B:", len(B), "I:", len(I), "F:", len(F), "T:", len(T))
print("Demanda D:", D)

if not D:
    print("\n⚠️  La demanda válida D está vacía. Verifica consistencia entre orders.csv, plants.csv y simple_products.csv")

# ----------- Modelo -----------
model = LpProblem("Modelo_Logistico", LpMinimize)
MVAL = 1e6
delta = 10

Q = {(m, p, i): LpVariable(f"Q_{m}_{p}_{i}", lowBound=0) 
    for m in M for p in P for i in I}
QB = {(b, p, i): LpVariable(f"QB_{b}_{p}_{i}", lowBound=0) for b in B for p in P for i in I}
X = {(m, i, f, t): LpVariable(f"X_{m}_{i}_{f}_{t}", lowBound=0) for m in M for i in I for f in F for t in T}
XB = {(b, i, f, t): LpVariable(f"XB_{b}_{i}_{f}_{t}", lowBound=0) for b in B for i in I for f in F for t in T}
Z = {(i, f): LpVariable(f"Z_{i}_{f}", cat=LpBinary) for i in I for f in F}

# ----------- Función Objetivo -----------
model += (
    lpSum((CAdq.get((p, m), 0) + CEnv.get((p, i), 0)) * Q[m, p, i] for m in M for p in P for i in I) +
    lpSum((CAdq.get((p, b), 0) + CEnv.get((p, i), 0)) * QB[b, p, i] for b in B for p in P for i in I) +
    lpSum(
        (CST.get((i, f, t), 0) + Imp.get((i, t, f), 0) +
         CManiIntake.get((i, t), 0) + CManiPlanta.get((f, t), 0) +
         Reg.get((i, f), 0) * delta) * X[m, i, f, t]
        for m in M for i in I for f in F for t in T
    ) +
    lpSum(
        (CST.get((i, f, t), 0) + Imp.get((i, t, f), 0) +
         CManiIntake.get((i, t), 0) + CManiPlanta.get((f, t), 0) +
         Reg.get((i, f), 0) * delta) * XB[b, i, f, t]
        for b in B for i in I for f in F for t in T
    )
)

# ----------- Restricciones de demanda -----------
for f in F:
    for m in M:
        if (f, m) in D:
            model += (
                lpSum(X[m, i, f, t] for i in I for t in T) +
                lpSum(XB[b, i, f, t] * B_prop.get((b, m), 0) for b in B for i in I for t in T)
                >= D[(f, m)]
            )

# ----------- Resolver -----------
orders_df = dataframes["orders"]
plants_df = dataframes["plants"]
products_df = dataframes["simple_products"]

print("PlantIDs en orders.csv no encontrados en plants.csv:")
print(set(orders_df["PlantID"]) - set(plants_df["PlantID"]))

print("SimpleProductIDs en orders.csv no encontrados en simple_products.csv:")
print(set(orders_df["SimpleProductID"]) - set(products_df["SimpleProductID"]))

model.solve()
print("\nEstado:", LpStatus[model.status])
print("Costo total:", value(model.objective))

# ----------- Mostrar variables activas -----------
print("\nVariables activas (valor > 0):")
for v in model.variables():
    if v.varValue and v.varValue > 0:
        print(f"{v.name} = {v.varValue}")
