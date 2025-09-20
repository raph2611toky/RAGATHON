import re

def strings(filename, min_length=4):
    with open(filename, "rb") as f:
        data = f.read()
    # On cherche les suites de caractères imprimables ASCII
    return re.findall(rb"[ -~]{%d,}" % min_length, data)

if __name__ == "__main__":
    fichier = "page28_graph3.png"
    for s in strings(fichier):
        try:
            print(s.decode("utf-8"))
        except UnicodeDecodeError:
            print(s.decode("latin-1"))
