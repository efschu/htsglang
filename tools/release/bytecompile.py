import subprocess, sys
root = sys.argv[1]
files = [p for p in subprocess.run(["git", "-C", root, "ls-files", "-z", "*.py"], capture_output=True, check=True).stdout.decode().split("\0") if p]
bad = []
for p in files:
    try:
        with open(f"{root}/{p}", "rb") as f:
            compile(f.read(), p, "exec", dont_inherit=True)
    except SyntaxError as e:
        bad.append(f"{p}: {e}")
print(f"bytecompile: {len(files)} .py, {len(bad)} errors"); print("\n".join(bad[:20]))
