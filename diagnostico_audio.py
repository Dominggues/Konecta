"""
Diagnóstico de áudio — rode isso antes do transcrever.py.
Mostra todos os dispositivos de loopback disponíveis e o volume
capturado em tempo real para confirmar que o áudio está chegando.
"""
import time
import soundcard as sc
import numpy as np

# 1. Lista todos os microfones (incluindo loopbacks)
print("=== Dispositivos de gravação disponíveis ===")
for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
    print(f"  [{i}] {mic.name}")

print()

# 2. Mostra o dispositivo que o transcrever.py usaria
alto_falante = sc.default_speaker()
print(f"=== Alto-falante padrão: {alto_falante.name} ===")
try:
    loopback = sc.get_microphone(id=str(alto_falante.name), include_loopback=True)
    print(f"Loopback encontrado: {loopback.name}")
except Exception as e:
    print(f"ERRO ao abrir loopback: {e}")
    print("Tente usar o índice de outro dispositivo da lista acima.")
    exit(1)

# 3. Captura 5 segundos e mostra o volume em tempo real
print("\n=== Reproduza algo no computador — monitorando volume por 5s ===")
SAMPLE_RATE = 16000
BLOCK = 4800

with loopback.recorder(samplerate=SAMPLE_RATE) as mic:
    inicio = time.time()
    while time.time() - inicio < 5:
        data = mic.record(numframes=BLOCK)
        mono = data[:, 0] if data.ndim > 1 else data
        rms = float(np.sqrt(np.mean(mono ** 2)))
        barra = "#" * int(rms * 500)
        print(f"  RMS: {rms:.4f}  {barra}", flush=True)

print("\nSe todos os RMS foram 0.0000, o loopback não está capturando áudio.")
print("Se os RMS subiram enquanto algo tocava, o transcrever.py deve funcionar.")
