"""
Modelo de optimización para supply chain de bundles.
Lee archivos CSV, estructura datos y ejecuta el modelo.
"""

import os
import sys
import pandas as pd
from pulp import (
    LpVariable, LpProblem, LpMinimize, lpSum, LpInteger, LpStatus, value,
    PulpError,
)

try:
    from natsort import natsorted
    sort_key = natsorted
except ImportError:
    sort_key = sorted

BASE_FOLDER = '.'
carpetas = [
    n for n in os.listdir(BASE_FOLDER)
    if os.path.isdir(os.path.join(BASE_FOLDER, n))
]
carpetas = sort_key(carpetas)

if not carpetas:
    print("No se encontraron carpetas de casos de prueba.")
    sys.exit(1)

print("=== Casos de prueba disponibles ===")
for i, carpeta in enumerate(carpetas, 1):
    print(f"{i}) {carpeta}")

while True:
    try:
        seleccion = int(input("\nSeleccione el número del caso: "))
        if 1 <= seleccion <= len(carpetas):
            folder = os.path.join(BASE_FOLDER, carpetas[seleccion - 1])
            break
        print(f"Ingrese un número entre 1 y {len(carpetas)}")
    except ValueError:
        print("Ingrese un número válido.")

print(f"\nUsando la carpeta: {folder}\n")

archivos = [
    'Suppliers.csv', 'Plants.csv', 'Receptions.csv', 'Bundles.csv',
    'Simple_products.csv', 'Simple_per_bundle.csv', 'Distances.csv',
    'Surcharge.csv', 'Orders.csv', 'Multimodal_transport.csv',
    'Route_supplier.csv'
]
data, errores, faltan = {}, [], []
for fname in archivos:
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        faltan.append(fname)
        continue
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except (ValueError, TypeError):
                pass
        data[fname] = df
    except (pd.errors.ParserError, FileNotFoundError) as e:
        errores.append(f"Error cargando {fname}: {e}")

if faltan:
    print("\n=== FALTAN ARCHIVOS OBLIGATORIOS ===")
    for f in faltan:
        print(f"  - {f}")
    sys.exit("\nAgrega los archivos faltantes y vuelve a ejecutar el script.\n")

if errores:
    print("\n=== ERRORES AL CARGAR ARCHIVOS ===")
    for err in errores:
        print("  -", err)
    sys.exit("\nCorrige los errores y vuelve a ejecutar el script.\n")

# Normalización nombres columnas
data['Suppliers.csv'] = data['Suppliers.csv'].rename(columns={
    'SupplierID': 'supplier_id', 'RegionID': 'supplier_region',
    'ProductProduced': 'product_id', 'ProductAvailability': 'availability',
    'PerUnitProductCost': 'unit_cost'
})
data['Plants.csv'] = data['Plants.csv'].rename(columns={
    'PlantID': 'plant_id', 'RegionID': 'plant_region',
    'PlantHandlingCostPerUnitAutomatic': 'cm_plant_auto',
    'PlantHandlingCostPerUnitManual': 'cm_plant_manual'
})
data['Receptions.csv'] = data['Receptions.csv'].rename(columns={
    'ReceptionID': 'reception_id', 'RegionID': 'reception_region',
    'ReceptionHandlingCostPerUnitAutomatic': 'cm_intake_auto',
    'ReceptionHandlingCostPerUnitManual': 'cm_intake_manual',
    'IsAutomaticCompatible': 'is_auto_compat'
})
data['Bundles.csv'] = data['Bundles.csv'].rename(columns={
    'BundleID': 'bundle_id', 'IsAutomaticCompatible': 'is_auto_compat',
    'UnitWeightKilogram': 'weight'
})
data['Simple_products.csv'] = data['Simple_products.csv'].rename(columns={
    'SimpleProductID': 'product_id', 'IsAutomaticCompatible': 'is_auto_compat',
    'UnitWeightKilogram': 'weight'
})
data['Simple_per_bundle.csv'] = data['Simple_per_bundle.csv'].rename(columns={
    'BundleID': 'bundle_id', 'SimpleProductID': 'product_id',
    'UnitsInBundle': 'quantity'
})
data['Distances.csv'] = data['Distances.csv'].rename(columns={
    'PlantID': 'plant_id', 'ReceptionID': 'reception_id',
    'DistanceKilometers': 'distance'
})
data['Surcharge.csv'] = data['Surcharge.csv'].rename(columns={
    'SupplierRegion': 'supplier_region', 'PlantRegion': 'plant_region',
    'TaxPerUnit': 'tax'
})
data['Orders.csv'] = data['Orders.csv'].rename(columns={
    'OrderID': 'order_id', 'PlantID': 'plant_id',
    'SimpleProductID': 'product_id', 'NumSimpleProductsOrdered': 'quantity'
})
data['Multimodal_transport.csv'] = data['Multimodal_transport.csv'].rename(
    columns={
        'ReceptionID': 'reception_id', 'PlantID': 'plant_id',
        'TransportationCostPerUnitAutomatic': 'ct_auto',
        'TransportationCostPerUnitManual': 'ct_manual',
        'TransportationCostPerUnitPremium': 'ct_premium',
        'HasHighway': 'has_highway',
        'HighwayTaxPerWeightKilogram': 'tax_highway'
    }
)
data['Route_supplier.csv'] = data['Route_supplier.csv'].rename(columns={
    'SupplierID': 'supplier_id', 'ReceptionID': 'reception_id',
    'TransportationCostPerUnit': 'ct_supplier_intake'
})

try:
    sup = data['Suppliers.csv']
    plt = data['Plants.csv']
    rec = data['Receptions.csv']
    bdl = data['Bundles.csv']
    sp = data['Simple_products.csv']
    spb = data['Simple_per_bundle.csv']
    dst = data['Distances.csv']
    srh = data['Surcharge.csv']
    ords = data['Orders.csv']
    mmd = data['Multimodal_transport.csv']
    rs = data['Route_supplier.csv']
except KeyError as e:
    sys.exit("Error en la estructura de los archivos de entrada: " + str(e))

def check_positive_integer(df_input, colnames, file):
    """
    Valida que en un DataFrame las columnas indicadas sean enteros positivos.
    """
    for colname in colnames:
        if colname not in df_input.columns:
            continue
        for idx, cell_value in df_input[colname].items():
            fila_excel = idx + 2
            try:
                val_num = float(cell_value)
            except ValueError:
                errores.append(
                    f"[{file}] Fila {fila_excel}, columna '{colname}': "
                    f"valor no numérico ('{cell_value}')"
                )
                continue
            if pd.isnull(val_num):
                errores.append(
                    f"[{file}] Fila {fila_excel}, columna '{colname}': valor nulo"
                )
            elif val_num < 0:
                errores.append(
                    f"[{file}] Fila {fila_excel}, columna '{colname}': valor negativo ({val_num})"
                )
            elif isinstance(val_num, float) and not val_num.is_integer():
                errores.append(
                    f"[{file}] Fila {fila_excel}, columna '{colname}': valor decimal ({val_num})"
                )

validaciones = {
    'Suppliers.csv': ['availability', 'unit_cost'],
    'Simple_per_bundle.csv': ['quantity'],
    'Orders.csv': ['quantity'],
    'Receptions.csv': [
        'ReceptionMaxNumOfUnitsAutomatic', 'ReceptionMaxTotalWeightAutomatic',
        'ReceptionMaxNumOfUnitsManual', 'ReceptionMaxTotalWeightManual',
        'cm_intake_auto', 'cm_intake_manual', 'is_auto_compat'
    ],
    'Plants.csv': ['cm_plant_auto', 'cm_plant_manual'],
    'Bundles.csv': ['weight', 'is_auto_compat'],
    'Simple_products.csv': ['weight', 'is_auto_compat'],
    'Distances.csv': ['distance'],
    'Surcharge.csv': ['tax'],
    'Multimodal_transport.csv': [
        'ct_auto', 'ct_manual', 'ct_premium', 'has_highway', 'tax_highway'
    ],
    'Route_supplier.csv': ['ct_supplier_intake'],
}

for archivo, columnas in validaciones.items():
    if archivo in data:
        check_positive_integer(data[archivo], columnas, archivo)

if errores:
    print("\n=== ERRORES DE VALIDACIÓN EN ARCHIVOS DE ENTRADA ===")
    for err in errores:
        print(err)
    sys.exit("\nCorrige los errores anteriores y vuelve a ejecutar el script.\n")

print("\n==== MUESTRA DE LOS DATOS UTILIZADOS EN EL MODELO ====\n")
for fname, df in data.items():
    print(f"Archivo: {fname}")
    if not df.empty:
        print(df.head(3).to_string(index=False))
    else:
        print("  (Archivo vacío)")
    print("-" * 55)
P=sup['supplier_id'].unique()
K=plt['plant_id'].unique()
R=rec['reception_id'].unique()
C=bdl['bundle_id'].unique()
M=sp['product_id'].unique()
T=['auto','manual','premium']
P_C = [
    (row['supplier_id'], row['product_id'])
    for idx, row in sup[sup['product_id'].isin(C)].iterrows()
]
cost_pc = (
    sup[sup['product_id'].isin(C)]
    .set_index(['supplier_id', 'product_id'])['unit_cost']
    .to_dict()
)
q_pcm=spb.set_index(['bundle_id','product_id'])['quantity'].to_dict()
surcharge=srh.set_index(['supplier_region','plant_region'])['tax'].to_dict()
ct_supplier_intake=rs.set_index(['supplier_id','reception_id'])['ct_supplier_intake'].to_dict()
ct={}
for _,row in mmd.iterrows():
    ct[(row['reception_id'],row['plant_id'],'auto')]=row['ct_auto']
    ct[(row['reception_id'],row['plant_id'],'manual')]=row['ct_manual']
    ct[(row['reception_id'],row['plant_id'],'premium')]=row['ct_premium']
has_highway=mmd.set_index(['reception_id','plant_id'])['has_highway'].to_dict()
tax_highway=mmd.set_index(['reception_id','plant_id'])['tax_highway'].to_dict()
cm_intake=rec.set_index('reception_id')['cm_intake_auto'].to_dict()
cm_plant=plt.set_index('plant_id')['cm_plant_auto'].to_dict()
proveedor_region=sup.set_index('supplier_id')['supplier_region'].to_dict()
reception_region=rec.set_index('reception_id')['reception_region'].to_dict()
demand=ords.set_index(['plant_id','product_id'])['quantity'].to_dict()
is_auto_compat_bundle=bdl.set_index('bundle_id')['is_auto_compat'].to_dict()
is_auto_compat_reception=rec.set_index('reception_id')['is_auto_compat'].to_dict()
cap_auto=rec.set_index('reception_id')['ReceptionMaxNumOfUnitsAutomatic'].to_dict()
cap_manual=rec.set_index('reception_id')['ReceptionMaxNumOfUnitsManual'].to_dict()
b=LpVariable.dicts('b',(P_C,R),lowBound=0,
                   cat=LpInteger)
y=LpVariable.dicts('y',(R,K,C,T),lowBound=0,
                   cat=LpInteger)
modelo=LpProblem("SupplyChain",LpMinimize)
modelo += (
    lpSum(
        b[(p, c)][r] * cost_pc.get((p, c), 0)
        for (p, c) in P_C for r in R
    )
    + lpSum(
        b[(p, c)][r] * ct_supplier_intake.get((p, r), 0)
        for (p, c) in P_C for r in R
    )
    + lpSum(
        y[r][k][c][t] * ct.get((r, k, t), 0)
        for r in R for k in K for c in C for t in T
    )
    + lpSum(
        b[(p, c)][r] * surcharge.get(
            (proveedor_region.get(p), reception_region.get(r)), 0
        )
        for (p, c) in P_C for r in R
    )
    + lpSum(
        b[(p, c)][r] * cm_intake.get(r, 0)
        for (p, c) in P_C for r in R
    )
    + lpSum(
        y[r][k][c][t] * cm_plant.get(k, 0)
        for r in R for k in K for c in C for t in T
    )
    + lpSum(
        y[r][k][c]['premium'] * has_highway.get((r, k), 0)
        * tax_highway.get((r, k), 0)
        for r in R for k in K for c in C
    )
), "Costo_total_sistema"
for k in K:
    for m in M:
        modelo+=(
            lpSum(y[r][k][c][t]*q_pcm.get((c,m),0) for r in R for c in C for t in T)>=
            demand.get((k,m),0),f"demanda_planta_{k}_{m}"
        )
for r in R:
    for c in C:
        modelo+=(
            lpSum(b[(p2,c2)][r] for (p2,c2) in P_C if c2==c)==
            lpSum(y[r][k][c][t] for k in K for t in T),f"balance_intake_{r}_{c}"
        )
for r in R:
    for k in K:
        for c in C:
            if not is_auto_compat_bundle.get(c,1) or not is_auto_compat_reception.get(r,1):
                modelo+=(lpSum(y[r][k][c]['auto'])==0,
                         f"compat_auto_{r}_{k}_{c}")
for r in R:
    modelo+=(lpSum(y[r][k][c]['auto'] for k in K for c in C)<=cap_auto.get(r,1e9),
            f"capacidad_max_auto_{r}")
    modelo+=(lpSum(y[r][k][c]['manual'] for k in K for c in C)<=cap_manual.get(r,1e9),
            f"capacidad_max_manual_{r}")
for (p,c) in P_C:
    total_comprado=lpSum(b[(p,c)][r] for r in R)
    max_disponible=sup.set_index(['supplier_id','product_id']).loc[(p,c),'availability']
    modelo+=(total_comprado<=max_disponible,f"restric_disponibilidad_{p}_{c}")
try:
    modelo.solve()
except PulpError as pe:
    print(f"[ERROR] Al resolver el modelo: {pe}")
    sys.exit(1)
except Exception as e:  # pylint: disable=broad-except
    print(f"[ERROR inesperado] {e}")
    sys.exit(1)
estado=LpStatus[modelo.status]
if estado=="Infeasible":
    for (plant_id,product_id),demanded in demand.items():
        TOTAL_SUPPLY = 0

        for (supplier_id,bundle_id) in P_C:
            bundle_qty=q_pcm.get((bundle_id,product_id),0)
            supplier_avail = sup.loc[
                (sup['supplier_id'] == supplier_id)
                & (sup['product_id'] == bundle_id),
                'availability'
            ].sum()
            TOTAL_SUPPLY+=bundle_qty*supplier_avail
        if TOTAL_SUPPLY<demanded:
            print(f"[CAUSA] No se puede cumplir la demanda de '{product_id}' "
                  f"en planta '{plant_id}':")
            print(f"  - Demanda requerida: {demanded}")
            print(f"  - Máximo posible según disponibilidad de bundles/proveedores: {TOTAL_SUPPLY}")
print("\n================ RESULTADOS DEL MODELO ================\n")
print(f"Estado de optimización : {estado}")
if estado!="Optimal":
    print("¡Atención! El modelo NO encontró una solución óptima. Revise los datos.")
print(f"Costo óptimo total     : {value(modelo.objective):.2f}\n")
print("PLAN DE COMPRAS (b > 0):")
HAY_COMPRAS = False
for (p, c) in P_C:
    for r in R:
        v = b[(p, c)][r].varValue
        if v and v > 0:
            print(f"  - Proveedor {p} | Combo {c} → Intake {r}: {int(v)}")
            HAY_COMPRAS = True
if not HAY_COMPRAS:
    print("  (sin compras asignadas)")

HAY_TRANSPORTE = False
for r in R:
    for k in K:
        for c in C:
            for t in T:
                v = y[r][k][c][t].varValue
                if v and v > 0:
                    print(
                        f"  - Intake {r} → Planta {k} | Combo {c} | Modo {t}: {int(v)}"
                    )
                    HAY_TRANSPORTE = True
if not HAY_TRANSPORTE:
    print("  (sin transporte asignado)")

print("\n=======================================================\n")
print("\n==== Auditoría detallada del costo total ====\n")
total_cost = 0
print("1) Costo de compra de bundles:")
for (p, c) in P_C:
    for r in R:
        v = b[(p, c)][r].varValue or 0
        if v > 0:
            val = v * cost_pc.get((p, c), 0)
            print(
                f"  - {v} x Bundle {c} de {p} "
                f"(unitario {cost_pc.get((p, c), 0)}) = {val}"
            )
            total_cost += val
print("\n2) Costo transporte proveedor → intake:")
for (p,c) in P_C:
    for r in R:
        v=b[(p,c)][r].varValue or 0
        if v>0:
            val=v*ct_supplier_intake.get((p,r),0)
            print(
                f"  - {v} x Proveedor {p} → Intake {r} "
                f"(unitario {ct_supplier_intake.get((p, r), 0)}) = {val}"
            )
            total_cost+=val
print("\n3) Costo transporte intake → planta:")
for r in R:
    for k in K:
        for c in C:
            for t in T:
                v=y[r][k][c][t].varValue or 0
                if v>0:
                    val=v*ct.get((r,k,t),0)
                    print(
                        f"  - {v} x Intake {r} → Planta {k}, Bundle {c}, modo {t} "
                        f"(unitario {ct.get((r, k, t), 0)}) = {val}"
                    )
                    total_cost+=val
print("\n4) Surcharge (impuestos interregionales):")
for (p,c) in P_C:
    for r in R:
        v=b[(p,c)][r].varValue or 0
        tax=surcharge.get((proveedor_region.get(p),reception_region.get(r)),0)
        if v>0 and tax>0:
            val=v*tax
            print(
                f"  - {v} x {p} (región {proveedor_region.get(p)}) → {r} "
                f"(región {reception_region.get(r)}), unitario {tax} = {val}"
            )
            total_cost+=val
print("\n5) Handling en intake:")
for (p,c) in P_C:
    for r in R:
        v=b[(p,c)][r].varValue or 0
        cost=cm_intake.get(r,0)
        if v>0 and cost>0:
            val=v*cost
            print(f"  - {v} x Intake {r} (unitario {cost}) = {val}")
            total_cost+=val
print("\n6) Handling en planta:")
for r in R:
    for k in K:
        for c in C:
            for t in T:
                v=y[r][k][c][t].varValue or 0
                cost=cm_plant.get(k,0)
                if v>0 and cost>0:
                    val=v*cost
                    print(f"  - {v} x Planta {k} (unitario {cost}) = {val}")
                    total_cost+=val
print("\n7) Impuesto autopista (solo premium):")
for r in R:
    for k in K:
        for c in C:
            v=y[r][k][c]['premium'].varValue or 0
            if v>0:
                hh=has_highway.get((r,k),0)
                th=tax_highway.get((r,k),0)
                val=v*hh*th
                print(f"  - {v} x {r}→{k}, Bundle {c}, premium: {val}")
                total_cost+=val
print("\n===============================================")
print(f"TOTAL AUDITADO (suma de componentes): {total_cost:.2f}")
print(f"TOTAL OPTIMIZADO (solver): {value(modelo.objective):.2f}\n")
