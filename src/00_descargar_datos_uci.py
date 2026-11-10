from pathlib import Path
from ucimlrepo import fetch_ucirepo

root = Path(__file__).resolve().parents[1]
output = root / "data" / "raw" / "student_dropout_success.csv"

output.parent.mkdir(parents=True, exist_ok=True)

dataset = fetch_ucirepo(id=697)

X = dataset.data.features.copy()
y = dataset.data.targets.squeeze()

df = X.copy()
df["Target"] = y.to_numpy()

assert df.shape == (4424, 37)
assert df["Target"].notna().all()
assert set(df["Target"]) == {
    "Dropout", "Enrolled", "Graduate"
}

if output.exists():
    print("El dataset ya existe; no se sobrescribira.")
else:
    df.to_csv(output, index=False)
    print("Dataset recuperado desde UCI.")

print("Archivo:", output)
print("Registros:", len(df))
print("Variables predictoras:", X.shape[1])
