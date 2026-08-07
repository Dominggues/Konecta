"""
Captura o audio de saida do sistema (loopback - o que esta tocando na call do
Teams/Meet) em stream continuo, detecta os trechos de fala com VAD e manda
cada trecho para o faster-whisper transcrever, imprimindo a legenda assim
que a pessoa termina de falar (ou a cada poucos segundos, se ela falar sem
pausar).

Arquitetura (produtor/consumidor), pensada para latencia baixa:

  [thread] captura loopback  -->  fila de frames de 30ms  -->
  [thread] segmentador VAD   -->  fila de trechos de fala  -->
  [thread] transcricao (faster-whisper)  -->  print da legenda

Nada e' salvo em disco durante o streaming: tudo fica em memoria (numpy)
ate virar texto.
"""

import collections
import queue
import sys
import threading
import time

import numpy as np
import soundcard as sc
import webrtcvad
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # faster-whisper e webrtcvad trabalham em 16kHz
FRAME_MS = 30                # tamanho de cada frame lido do microfone (ms)
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD_AGRESSIVIDADE = 2        # 0 (mais permissivo) a 3 (mais rigoroso p/ ruido)
JANELA_DECISAO_MS = 300      # quantos ms de historico o VAD olha p/ decidir
FRAMES_JANELA = JANELA_DECISAO_MS // FRAME_MS

MAX_SEGMENTO_S = 8           # forca o envio do trecho mesmo sem silencio,
                              # para nao esperar demais numa fala longa

MODELO_WHISPER = "small"     # tiny/base/small/medium/large-v3 (trade-off velocidade x precisao)
DEVICE = "cpu"                # troque para "cuda" se tiver GPU NVIDIA
COMPUTE_TYPE = "int8"         # int8 = mais rapido em CPU; "float16" se usar GPU
IDIOMA = "pt"

# ---------------------------------------------------------------------------
# Filas entre as threads
# ---------------------------------------------------------------------------
fila_frames = queue.Queue()
fila_segmentos = queue.Queue()
parar = threading.Event()


def capturar_audio():
    """Le o loopback do alto-falante padrao em pedacinhos de FRAME_MS e
    empilha na fila_frames, ja convertido para int16 mono (formato que o
    webrtcvad exige)."""
    alto_falante = sc.default_speaker()
    mic_loopback = sc.get_microphone(id=str(alto_falante.name), include_loopback=True)
    print(f"[captura] usando loopback de: {alto_falante.name}")

    with mic_loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
        while not parar.is_set():
            dados = mic.record(numframes=FRAME_SAMPLES)  # float32 [-1, 1], shape (N, 1)
            mono = dados[:, 0]
            pcm16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
            fila_frames.put(pcm16.tobytes())

    fila_frames.put(None)  # sinaliza fim


def segmentar_fala():
    """Consome frames de 30ms, decide onde a fala comeca/termina (VAD) e
    manda o trecho completo (em float32) para a fila de transcricao assim
    que detecta uma pausa - isso e' o que da a sensacao de "quase
    instantaneo": nao esperamos a call inteira, so a frase."""
    vad = webrtcvad.Vad(VAD_AGRESSIVIDADE)
    ring_buffer = collections.deque(maxlen=FRAMES_JANELA)
    em_fala = False
    segmento = []
    inicio_segmento = None

    while True:
        frame = fila_frames.get()
        if frame is None:
            break

        voz = vad.is_speech(frame, SAMPLE_RATE)

        if not em_fala:
            ring_buffer.append((frame, voz))
            num_voz = len([f for f, v in ring_buffer if v])
            if num_voz > 0.9 * ring_buffer.maxlen:
                em_fala = True
                inicio_segmento = time.time()
                segmento.extend(f for f, _ in ring_buffer)
                ring_buffer.clear()
        else:
            segmento.append(frame)
            ring_buffer.append((frame, voz))
            num_silencio = len([f for f, v in ring_buffer if not v])
            duracao = time.time() - inicio_segmento

            if num_silencio > 0.9 * ring_buffer.maxlen or duracao > MAX_SEGMENTO_S:
                pcm = b"".join(segmento)
                audio_np = (
                    np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                )
                fila_segmentos.put(audio_np)
                segmento = []
                ring_buffer.clear()
                em_fala = False

    if segmento:
        pcm = b"".join(segmento)
        audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        fila_segmentos.put(audio_np)

    fila_segmentos.put(None)


def transcrever():
    print(f"[modelo] carregando faster-whisper '{MODELO_WHISPER}' ({DEVICE}/{COMPUTE_TYPE})...")
    modelo = WhisperModel(MODELO_WHISPER, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("[modelo] pronto. Fale ou toque audio na call para ver a legenda.\n")

    while True:
        audio_np = fila_segmentos.get()
        if audio_np is None:
            break
        if len(audio_np) < SAMPLE_RATE * 0.3:  # ignora ruidos curtos (<300ms)
            continue

        t0 = time.time()
        segmentos, _info = modelo.transcribe(
            audio_np,
            language=IDIOMA,
            beam_size=1,        # beam pequeno = mais rapido, um pouco menos preciso
            vad_filter=False,   # ja segmentamos com webrtcvad antes
        )
        texto = "".join(s.text for s in segmentos).strip()
        if texto:
            latencia = time.time() - t0
            print(f"[legenda] {texto}   (transcrito em {latencia:.2f}s)")


def main():
    threads = [
        threading.Thread(target=capturar_audio, daemon=True),
        threading.Thread(target=segmentar_fala, daemon=True),
        threading.Thread(target=transcrever, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[main] encerrando...")
        parar.set()
        for t in threads:
            t.join(timeout=2)


if __name__ == "__main__":
    main()
