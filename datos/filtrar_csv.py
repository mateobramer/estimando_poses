"""
Filtra train.csv, val.csv y test.csv para quedarse solo con las filas
cuya imagen realmente existe en la carpeta frames/.

USO:
    python filtrar_csv.py --dataset dataset_final

SALIDA:
    Sobreescribe train.csv, val.csv, test.csv con solo las filas validas.
    Guarda los originales como train_original.csv, val_original.csv, test_original.csv
"""

import argparse
import os
import pandas as pd

def filtrar(csv_path, frames_dir):
    df = pd.read_csv(csv_path)
    total = len(df)
    
    existe = df["imagen"].apply(
        lambda f: os.path.isfile(os.path.join(frames_dir, f))
    )
    df_valido = df[existe].reset_index(drop=True)
    validos = len(df_valido)
    
    # Guardar original como backup
    backup = csv_path.replace(".csv", "_original.csv")
    os.rename(csv_path, backup)
    print(f"  Backup guardado: {backup}")
    
    # Guardar filtrado
    df_valido.to_csv(csv_path, index=False)
    print(f"  {csv_path}: {total} filas -> {validos} validas ({total-validos} eliminadas)")
    
    return validos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_final",
                        help="carpeta raiz del dataset (debe tener frames/, train.csv, val.csv)")
    args = parser.parse_args()

    frames_dir = os.path.join(args.dataset, "frames")
    
    if not os.path.isdir(frames_dir):
        print(f"No encontro la carpeta: {frames_dir}")
        return

    print(f"Filtrando CSVs en '{args.dataset}' contra imagenes en '{frames_dir}'...")
    print()

    total_validos = 0
    for nombre in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(args.dataset, nombre)
        if os.path.exists(csv_path):
            total_validos += filtrar(csv_path, frames_dir)
        else:
            print(f"  {nombre}: no encontrado, saltando")

    print(f"\nListo. Total de filas validas entre todos los splits: {total_validos}")
    print("Podés volver a subir los CSVs filtrados a la VM y entrenar.")

if __name__ == "__main__":
    main()
