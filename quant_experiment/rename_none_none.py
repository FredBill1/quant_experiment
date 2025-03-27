from .config import MODEL_PATH

path = MODEL_PATH.parent / "search"
files = list(path.glob("None_None*"))
for file in files:
    print(file)
    file.rename(file.with_name(file.name.replace("None_None", "baseline")))
