"""
Transcrição em tempo real do áudio do sistema (reunião, vídeo, etc.).
Captura o que está tocando no computador via loopback e envia para
a API do Google Speech (requer internet).

Uso:
    python transcrever.py

Ctrl+C para parar.
"""
import queue
import threading

import numpy as np
import soundcard as sc
import speech_recognition as sr

SAMPLE_RATE     = 16_000    # Hz
BLOCK_SIZE      = 4800      # ~300 ms por bloco capturado
IDIOMA          = "pt-BR"

# Detecção de fala por energia (VAD simples)
# Aumente RMS_FALA se capturar silêncio como fala (ruído de fundo)
RMS_FALA        = 0.05      # ajustado para o nível real de silêncio detectado
SILENCIO_FRAMES = 3         # blocos de silêncio para encerrar frase (~900 ms)
FALA_MIN_FRAMES = 2         # mínimo de blocos com fala para transcrever (~600 ms)
MAX_FALA_FRAMES = 30        # força transcrição após ~9 s de fala contínua sem pausa

_fila: queue.Queue = queue.Queue()
_reconhecedor = sr.Recognizer()


def _transcrever(audio: np.ndarray) -> None:
    """Envia trecho de fala para Google Speech e imprime o texto."""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)
    try:
        texto = _reconhecedor.recognize_google(audio_data, language=IDIOMA)
        if texto:
            print(texto, flush=True)
    except sr.UnknownValueError:
        pass                          # trecho incompreensível, ignora
    except sr.RequestError as e:
        print(f"[Erro de conexão: {e}]", flush=True)


def _loop() -> None:
    """Acumula blocos e dispara transcrição ao detectar pausa na fala."""
    buf: list[np.ndarray] = []
    n_silencio = 0
    n_fala = 0

    while True:
        bloco = _fila.get()
        rms = float(np.sqrt(np.mean(bloco ** 2)))

        if rms >= RMS_FALA:
            buf.append(bloco)
            n_fala += 1
            n_silencio = 0

            # Sem pausa por muito tempo: transcreve o que acumulou e continua
            if n_fala >= MAX_FALA_FRAMES:
                audio = np.concatenate(buf).astype(np.float32)
                threading.Thread(target=_transcrever, args=(audio,), daemon=True).start()
                buf.clear()
                n_fala = 0
        elif buf:
            buf.append(bloco)
            n_silencio += 1

            if n_silencio >= SILENCIO_FRAMES:
                if n_fala >= FALA_MIN_FRAMES:
                    audio = np.concatenate(buf).astype(np.float32)
                    # Thread separada: captura não para enquanto Google processa
                    threading.Thread(target=_transcrever, args=(audio,), daemon=True).start()
                buf.clear()
                n_fala = 0
                n_silencio = 0


# Descobre o alto-falante padrão e abre o loopback dele
alto_falante = sc.default_speaker()
print(f"Dispositivo: {alto_falante.name}")
print("Reproduza a reunião ou vídeo. Ctrl+C para parar.\n")

threading.Thread(target=_loop, daemon=True).start()

loopback = sc.get_microphone(id=str(alto_falante.name), include_loopback=True)

with loopback.recorder(samplerate=SAMPLE_RATE) as mic:
    try:
        while True:
            data = mic.record(numframes=BLOCK_SIZE)
            # soundcard retorna (frames, canais) — converte para mono
            mono = data[:, 0] if data.ndim > 1 else data
            _fila.put(mono)
    except KeyboardInterrupt:
        print("\nParado.")
