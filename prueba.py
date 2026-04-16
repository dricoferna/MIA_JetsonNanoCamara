import sys
import platform

print("✅ PROYECTO OK EN JETSON NANO")
print("--------------------------------")
print("Python version:", sys.version)
print("Sistema:", platform.system())
print("Arquitectura:", platform.machine())

# Test básico de cálculo
a = 5
b = 7
print("Test matemático:", a * b)

print("🎉 Todo funciona correctamente")