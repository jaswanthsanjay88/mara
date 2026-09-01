import jupyter_kernel_client as m

print("version:", getattr(m, "__version__", "?"))
print("attrs:", [a for a in dir(m) if not a.startswith("_")])
