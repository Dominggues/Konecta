import sys
import io as io_module

# Forca encoding UTF-8 no terminal Windows para evitar erros de acentos
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io_module.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io_module.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import soundcard as sc
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time
import wave
import io
import warnings

warnings.filterwarnings('ignore', category=sc.SoundcardRuntimeWarning)

# ============================================================
# Configuracoes Tecnicas
# ============================================================
MODELO_TAMANHO = "small"
TAXA_AMOSTRAGEM = 44100          # sample rate padrao do soundcard/WASAPI
TEMPO_CHUNK = 0.3                # 300ms por bloco
THRESHOLD_VOLUME = 0.002         # RMS minimo para considerar "fala"
MAX_SILENCIO = 1.5               # segundos de silencio antes de fechar frase
MIN_AUDIO_DURATION = 0.5         # duracao minima (s) -- audio muito curto descarta
BUFFER_MAX_SAMPLES = 220500      # ~5s em 44.1kHz -- evita estouro de memoria

# ============================================================
# Estado global
# ============================================================
fila_audio = queue.Queue()
rodando = True


def capturar_audio_vad():
    """
    Captura audio do sistema via WASAPI loopback (Windows) e agrupa
    os blocos ate detectar silencio. Cada frase fechada vai para a fila.
    """
    global rodando

    while rodando:
        try:
            alto_falante = sc.default_speaker()
            loopback = sc.get_microphone(id=str(alto_falante.name), include_loopback=True)
            print(f"\n[Captura] Conectado em loopback: {alto_falante.name} ({loopback.channels}ch)")
            print("--- KONECTA: OUVINDO ---\n", flush=True)

            buffer_frase = []
            silencio_acumulado = 0.0
            frames = int(TAXA_AMOSTRAGEM * TEMPO_CHUNK)

            with loopback.recorder(samplerate=TAXA_AMOSTRAGEM) as mic:
                while rodando:
                    dados = mic.record(numframes=frames)

                    # Mono
                    if dados.ndim > 1:
                        dados = np.mean(dados, axis=1)
                    dados = dados.astype(np.float32)

                    volume = np.sqrt(np.mean(dados ** 2))
                    duracao_buffer = len(buffer_frase) / TAXA_AMOSTRAGEM

                    # Evita buffer infinito (trava ~5s max)
                    if len(buffer_frase) > BUFFER_MAX_SAMPLES:
                        buffer_frase = buffer_frase[-BUFFER_MAX_SAMPLES:]

                    if volume > THRESHOLD_VOLUME:
                        # Tem voz -- guarda e zera relogio de silencio
                        buffer_frase.extend(dados)
                        silencio_acumulado = 0.0
                    else:
                        silencio_acumulado += TEMPO_CHUNK

                        # Silencio suficiente + buffer tem conteudo? Fecha a frase.
                        if (silencio_acumulado >= MAX_SILENCIO
                                and duracao_buffer >= MIN_AUDIO_DURATION):
                            np_buffer = np.array(buffer_frase, dtype=np.float32)
                            buffer_frase = []
                            silencio_acumulado = 0.0
                            fila_audio.put((time.time(), np_buffer))
                            print(
                                f"  >> Clip enviado ({duracao_buffer:.1f}s)",
                                flush=True,
                            )

        except OSError as e:
            print(f"[Captura] Dispositivo inacessivel: {e} - reconectando...", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"[Captura] Erro: {e}", flush=True)
            time.sleep(1)


def _numpy_to_wav(audio_np, sr=16000):
    """Converte numpy array (float32, mono) em BytesIO WAV para o Whisper."""
    buf = io.BytesIO()
    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf


def _resample(audio, de_sr, para_sr=16000):
    """Downsampling simples por media (evita dependencia extras)."""
    fator = de_sr // para_sr
    resto = len(audio) % fator
    trim = audio[:-resto] if resto else audio
    return trim.reshape(-1, fator).mean(axis=1).astype(np.float32)


def processar_transcricao():
    """
    Recebe clips fechados da fila, converte para 16kHz e transcreve
    com Whisper (small, CPU, int8).
    """
    global rodando

    print(f"[Transcricao] Carregando modelo '{MODELO_TAMANHO}'...")
    modelo = WhisperModel(MODELO_TAMANHO, device="cpu", compute_type="int8")
    print("[Transcricao] IA pronta.\n", flush=True)

    while rodando:
        try:
            timestamp, audio_np = fila_audio.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            # Downsample para 16 kHz (esperado pelo Whisper)
            audio_16k = _resample(audio_np, de_sr=TAXA_AMOSTRAGEM)
            wav_like = _numpy_to_wav(audio_16k)

            segmentos, info = modelo.transcribe(
                wav_like,
                language="pt",
                beam_size=5,
                vad_filter=True,
            )

            for seg in segmentos:
                texto = seg.text.strip()
                # So exibe se for realmente fala (anti-halucinacao)
                if texto and seg.no_speech_prob < 0.60:
                    ts = time.strftime("%H:%M:%S", time.localtime(timestamp))
                    print(f"[{ts}] {texto}", flush=True)

        except Exception as e:
            print(f"[Transcricao] Erro: {e}", flush=True)
        finally:
            fila_audio.task_done()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Iniciando pipeline de captacao e transcricao em tempo real...")
    print("Toque algo (audio, video) no PC para comecar a ouvir.\n", flush=True)

    t_cap = threading.Thread(target=capturar_audio_vad, daemon=True)
    t_trans = threading.Thread(target=processar_transcricao, daemon=True)

    t_cap.start()
    t_trans.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Sistema] Encerrando...")
        rodando = False
        t_cap.join(timeout=3)
        t_trans.join(timeout=3)
        print("[Sistema] Encerrado com sucesso.")
