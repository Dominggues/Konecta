"""
Passo extra necessario so' no Windows, so' se for usar GPU (DEVICE = "cuda"
em transcricao_tempo_real.py).

O motivo: no Windows, o ctranslate2 (motor por tras do faster-whisper) nao
encontra sozinho as DLLs de CUDA/cuDNN instaladas via pip (nvidia-cublas-cu12,
nvidia-cudnn-cu12 etc) mesmo com elas presentes no venv - ele so' enxerga
DLLs que estao na SUA PROPRIA pasta (site-packages/ctranslate2/). Esse script
copia as DLLs de la' pra ca'.

Rode uma vez, depois de "pip install -r requirements.txt":

    .venv\\Scripts\\python.exe configurar_gpu_windows.py

Se voce nao tem GPU NVIDIA, nao precisa rodar isso - o projeto cai pra CPU
automaticamente (veja o fallback em transcricao_tempo_real.py).
"""

import glob
import os
import shutil
import sys


def main():
    if os.name != "nt":
        print("Esse script e' so' para Windows. Nada a fazer.")
        return

    base = os.path.join(sys.prefix, "Lib", "site-packages")
    origem_nvidia = os.path.join(base, "nvidia")
    destino = os.path.join(base, "ctranslate2")

    if not os.path.isdir(origem_nvidia):
        print(
            "Pastas 'nvidia' nao encontradas em site-packages.\n"
            "Instale antes: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
            "nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12"
        )
        return

    if not os.path.isdir(destino):
        print(f"Pasta do ctranslate2 nao encontrada em {destino}. O pacote esta instalado?")
        return

    dlls = glob.glob(os.path.join(origem_nvidia, "*", "bin", "*.dll"))
    if not dlls:
        print("Nenhuma DLL encontrada dentro de nvidia/*/bin.")
        return

    copiadas = 0
    for dll in dlls:
        alvo = os.path.join(destino, os.path.basename(dll))
        shutil.copy2(dll, alvo)
        copiadas += 1
        print(f"copiado: {os.path.basename(dll)}")

    print(f"\n{copiadas} DLL(s) copiada(s) para {destino}")
    print("Pronto. Agora transcricao_tempo_real.py com DEVICE='cuda' deve funcionar.")


if __name__ == "__main__":
    main()
