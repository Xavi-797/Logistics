import pandas as pd
import os
from pulp import *

# ----------- Solicitar nombre de carpeta -----------
carpeta = input("🔍 Ingresa el nombre de la carpeta con los archivos CSV: ").strip()

if not os.path.isdir(carpeta):
    print(f"❌ La carpeta '{carpeta}' no existe. Verifica el nombre.")
    exit(1)

# ----------- Cargar archivos desde carpeta -----------
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
for clave, archivo in archivos.items():
    ruta = os.path.join(carpeta, archivo)
    if os.path.exists(ruta):
        try:
            df = pd.read_csv(ruta, sep=";")
            df.columns = df.columns.str.strip().str.lower()
            dataframes[clave] = df
        except Exception as e:
            print(f"⚠️ Error al leer {archivo}: {e}")
    else:
        print(f"📂 Archivo no encontrado, ignorando: {archivo}")

# ----------- Conjuntos -----------
def extraer_conjuntos(df):
    P = df["suppliers"]["supplierid"].dropna().unique().tolist()
    M = df["simple_products"]["simpleproductid"].dropna().unique().tolist()
    B = df["bundles"]["bundleid"].dropna().unique().tolist()
    I = df["receptions"]["receptionid"].dropna().unique().tolist()
    F = df["plants"]["plantid"].dropna().unique().tolist()
    T = [col.replace("transportationcostperunit", "")
         for col in df["multimodal_transport"].columns if "transportationcostperunit" in col]
    return P, M, B, I, F, T

P, M, B, I, F, T = extraer_conjuntos(dataframes)

# ----------- Parámetros -----------
def construir_parametros(df):
    CAdq = {(r["supplierid"], r["productproduced"]): r["perunitproductcost"] for _, r in df["suppliers"].iterrows()}
    CapProv = {(r["supplierid"], r["productproduced"]): r["productavailability"] for _, r in df["suppliers"].iterrows()}
    CEnv = {(r["supplierid"], r["receptionid"]): r["transportationcostperunit"] for _, r in df["route_supplier"].iterrows()}
    Comp_mt = {(r["simpleproductid"], t): int(r["isautomaticcompatible"]) if t == "Automatic" else 1
               for _, r in df["simple_products"].iterrows() for t in T}
    Comp_bt = {(r["bundleid"], t): int(r["isautomaticcompatible"]) if t == "Automatic" else 1
               for _, r in df["bundles"].iterrows() for t in T}
    B_prop = {(r["bundleid"], r["simpleproductid"]): r["unitsinbundle"] for _, r in df["simple_per_bundle"].iterrows()}

    orders_df = df["orders"]
    orders_df.columns = orders_df.columns.str.strip().str.lower()
    D_raw = {(r["plantid"], r["simpleproductid"]): r["numsimpleproductsordered"] for _, r in orders_df.iterrows()}

    Reg = {(r["receptionid"], r["plantid"]): int(r["distancekilometers"] <= 50) for _, r in df["distances"].iterrows()}
    alpha = {(r["receptionid"], r["plantid"]): int(r["distancekilometers"] <= 30) for _, r in df["distances"].iterrows()}
    CST = {(r["receptionid"], r["plantid"], t): r[f"transportationcostperunit{t.lower()}"] for _, r in df["multimodal_transport"].iterrows() for t in T}
    Imp = {(r["receptionid"], t, r["plantid"]): r["highwaytaxperweightkilogram"] if r["hashighway"] else 0 for _, r in df["multimodal_transport"].iterrows() for t in T}
    CManiIntake = {(r["receptionid"], t): r[f"receptionhandlingcostperunit{t.lower()}"] for _, r in df["receptions"].iterrows() for t in ["Automatic", "Manual"]}
    CManiPlanta = {(r["plantid"], t): r[f"planthandlingcostperunit{t.lower()}"] for _, r in df["plants"].iterrows() for t in ["Automatic", "Manual"]}

    return CAdq, CapProv, CEnv, Comp_mt, Comp_bt, B_prop, D_raw, Reg, alpha, CST, Imp, CManiIntake, CManiPlanta

CAdq, CapProv, CEnv, Comp_mt, Comp_bt, B_prop, D_raw, Reg, alpha, CST, Imp, CManiIntake, CManiPlanta = construir_parametros(dataframes)

CapTransporte = {"InHouse": 100000, "TR1": 50000, "TR2": 75000}
D = {k: v for k, v in D_raw.items() if k[0] in F and k[1] in M}

print("P:", len(P), "M:", len(M), "B:", len(B), "I:", len(I), "F:", len(F), "T:", len(T))

# ----------- Modelo -----------
model = LpProblem("Modelo_Logistico", LpMinimize)
MVAL = 1e6
delta = 10
epsilon = 0.001

Q = {(m, p, i): LpVariable(f"Q_{m}_{p}_{i}", lowBound=0) for m in M for p in P for i in I}
QB = {(b, p, i): LpVariable(f"QB_{b}_{p}_{i}", lowBound=0) for b in B for p in P for i in I}
X = {(m, i, f, t): LpVariable(f"X_{m}_{i}_{f}_{t}", lowBound=0) for m in M for i in I for f in F for t in T}
XB = {(b, i, f, t): LpVariable(f"XB_{b}_{i}_{f}_{t}", lowBound=0) for b in B for i in I for f in F for t in T}
Z = {(i, f): LpVariable(f"Z_{i}_{f}", cat=LpBinary) for i in I for f in F}

model += (
    lpSum((CAdq.get((p, m), 0) + CEnv.get((p, i), 0)) * Q[m, p, i] for m in M for p in P for i in I) +
    lpSum((CAdq.get((p, b), 0) + CEnv.get((p, i), 0)) * QB[b, p, i] for b in B for p in P for i in I) +
    lpSum((CST.get((i, f, t), 0) + Imp.get((i, t, f), 0) + CManiIntake.get((i, t), 0) + CManiPlanta.get((f, t), 0) + Reg.get((i, f), 0) * delta) * X[m, i, f, t] for m in M for i in I for f in F for t in T) +
    lpSum((CST.get((i, f, t), 0) + Imp.get((i, t, f), 0) + CManiIntake.get((i, t), 0) + CManiPlanta.get((f, t), 0) + Reg.get((i, f), 0) * delta) * XB[b, i, f, t] for b in B for i in I for f in F for t in T)
)

# ----------- Restricciones -----------
for f in F:
    for m in M:
        if (f, m) in D:
            model += (
                lpSum(X[m, i, f, t] for i in I for t in T) +
                lpSum(XB[b, i, f, t] * B_prop.get((b, m), 0) for b in B for i in I for t in T) >= D[(f, m)]
            )

for m in M:
    for i in I:
        for f in F:
            for t in T:
                model += X[m, i, f, t] <= Comp_mt.get((m, t), 0) * MVAL

for b in B:
    for i in I:
        for f in F:
            for t in T:
                model += XB[b, i, f, t] <= Comp_bt.get((b, t), 0) * MVAL

# 🚨 Balance intake - planta (nuevo)
for m in M:
    for i in I:
        model += (
            lpSum(X[m, i, f, t] for f in F for t in T) <=
            lpSum(Q[m, p, i] for p in P)
        )

for b in B:
    for i in I:
        model += (
            lpSum(XB[b, i, f, t] for f in F for t in T) <=
            lpSum(QB[b, p, i] for p in P)
        )

for p in P:
    for m in M:
        cap = CapProv.get((p, m), None)
        if cap is not None:
            model += (
                lpSum(Q[m, p, i] for i in I) +
                lpSum(QB[b, p, i] * B_prop.get((b, m), 0) for b in B for i in I)
            ) <= cap

for t in T:
    cap = CapTransporte.get(t, None)
    if cap is not None:
        model += (
            lpSum(X[m, i, f, t] for m in M for i in I for f in F) +
            lpSum(XB[b, i, f, t] for b in B for i in I for f in F)
        ) <= cap

for f in F:
    model += lpSum(Z[i, f] for i in I) >= 2
    model += lpSum(Z[i, f] for i in I) <= 5

for i in I:
    for f in F:
        model += (
            lpSum(X[m, i, f, t] for m in M for t in T) +
            lpSum(XB[b, i, f, t] for b in B for t in T)
        ) >= epsilon * Z[i, f]

# ----------- Resolver e imprimir resultados -----------
model.solve()
print("\nEstado:", LpStatus[model.status])
print("Costo total:", value(model.objective))

print("\nVariables activas (valor > 0):")
for v in model.variables():
    if v.varValue and v.varValue > 0:
        print(f"{v.name} = {v.varValue}")
