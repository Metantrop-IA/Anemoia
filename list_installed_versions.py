import tomli
import importlib.metadata
import re
import sys

def parse_package_name(dep):
    # Remove version specifiers and extras
    return re.split(r'[<>=!~\[]', dep.strip())[0]

def main():
    print(f"Python version: {sys.version.split()[0]}\n")
    with open("Anemoia/Anemoia/pyproject.toml", "rb") as f:
        data = tomli.load(f)
    deps = data["project"]["dependencies"]
    print("Dependency versions in current environment:\n")
    for dep in deps:
        pkg = parse_package_name(dep)
        try:
            version = importlib.metadata.version(pkg)
            print(f"{pkg}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")

if __name__ == "__main__":
    main()