"""
Captura o audio de saida do sistema (loopback - o que esta tocando na call do
Teams/Meet) em stream continuo e transcreve com legenda INCREMENTAL: em vez
de esperar a frase terminar, o trecho de fala e' retranscrito varias vezes
enquanto a pessoa ainda esta falando, e so exibimos as palavras que ja se
"estabilizaram" entre duas passadas seguidas (tecnica conhecida como local
agreement - e' basicamente como Teams/Meet fazem a legenda ao vivo deles).

Arquitetura:

  [thread] captura loopback  -->  fila de frames de 30ms  -->
  [thread] segmentacao (VAD) + transcricao incremental (faster-whisper)
                              -->  print da legenda (parcial e final)

A segmentacao e a transcricao ficam na mesma thread de proposito: cada
passada parcial e' rapida (GPU) e o buffer de frames continua enchendo em
paralelo na outra thread, entao nao ha perda de audio nem trava perceptivel.

Nada e' salvo em disco durante o streaming: tudo fica em memoria (numpy) ate
virar texto.
"""

import collections
import os
import queue
import sys
import threading
import time

# No Windows, o ctranslate2 (usado pelo faster-whisper) nao empacota as DLLs
# de CUDA (cuBLAS/cuDNN) junto - elas vem via pip a parte (nvidia-cublas-cu12,
# nvidia-cudnn-cu12) e o Windows precisa ser avisado onde procura-las. Isso
# tem que rodar ANTES de importar faster_whisper/ctranslate2, senao a
# primeira chamada de transcribe() quebra com "DLL not found" mesmo com a
# GPU disponivel.
if os.name == "nt":
    import glob

    _nvidia_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    for _dll_dir in glob.glob(os.path.join(_nvidia_dir, "*", "bin")):
        os.add_dll_directory(_dll_dir)
        os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

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
JANELA_DECISAO_MS = 300      # quantos ms de historico o VAD olha p/ decidir o INICIO da fala
FRAMES_JANELA = JANELA_DECISAO_MS // FRAME_MS

JANELA_FIM_MS = 200          # janela menor p/ decidir o FIM (corta mais rapido numa pausa curta)
FRAMES_JANELA_FIM = JANELA_FIM_MS // FRAME_MS
RATIO_INICIO_FALA = 0.9      # % de frames com voz p/ considerar que a fala comecou
RATIO_FIM_FALA = 0.7         # % de frames em silencio p/ considerar que a fala parou

# Com legenda incremental, o MAX_SEGMENTO_S deixa de ser o principal controle
# de latencia (quem faz isso agora e' PARTIAL_INTERVAL_S) - podemos deixar
# ele mais folgado pra dar mais contexto pro modelo na passada final, sem
# prejudicar a sensacao de velocidade.
MAX_SEGMENTO_S = 8

PARTIAL_INTERVAL_S = 0.6     # de quanto em quanto tempo refazemos a transcricao
                              # parcial enquanto a pessoa ainda esta falando
MIN_AUDIO_PARCIAL_S = 0.5    # nao vale a pena rodar o modelo com menos que isso

MODELO_WHISPER = "small"     # tiny/base/small/medium/large-v3 (trade-off velocidade x precisao)
                              # com GPU sobrando velocidade, "medium" costuma valer a pena
DEVICE = "cuda"               # "cuda" (GPU NVIDIA, muito mais rapido) ou "cpu" se nao tiver GPU
COMPUTE_TYPE = "float16"      # float16 = ideal p/ GPU; troque p/ "int8" se DEVICE="cpu"
IDIOMA = "pt"

# ---------------------------------------------------------------------------
# Fila entre as threads
# ---------------------------------------------------------------------------
fila_frames = queue.Queue()
parar = threading.Event()


def capturar_audio():
    """Le o loopback do alto-falante padrao em pedacinhos de FRAME_MS e
    empilha na fila_frames, ja convertido para int16 mono (formato que o
    webrtcvad exige)."""
    alto_falante = sc.default_speaker()
    mic_loopback = sc.get_microphone(id=str(alto_falante.name), include_loopback=True)
    print(f"[captura] usando loopback de: {alto_falante.name}", flush=True)

    with mic_loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
        while not parar.is_set():
            dados = mic.record(numframes=FRAME_SAMPLES)  # float32 [-1, 1], shape (N, 1)
            mono = dados[:, 0]
            pcm16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
            fila_frames.put(pcm16.tobytes())

    fila_frames.put(None)  # sinaliza fim


def _pcm_para_float(frames_bytes):
    pcm = b"".join(frames_bytes)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _prefixo_comum(a, b):
    """Quantas palavras do inicio sao identicas entre duas listas."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _transcrever(modelo, audio_np):
    segmentos, _info = modelo.transcribe(
        audio_np,
        language=IDIOMA,
        beam_size=1,        # beam pequeno = mais rapido, um pouco menos preciso
        vad_filter=False,   # ja segmentamos com webrtcvad
        condition_on_previous_text=False,  # evita "puxar" erro de uma passada pra outra
    )
    return "".join(s.text for s in segmentos).strip()


class LegendaAoVivo:
    """Cuida de imprimir a legenda de um segmento em andamento: sobrescreve
    a mesma linha do terminal enquanto o texto ainda e' provisorio e fecha
    com uma quebra de linha quando o trecho e' confirmado como final."""

    def __init__(self):
        self.confirmadas = 0     # quantas palavras ja foram "travadas"
        self.palavras = []       # palavras confirmadas ate agora, deste segmento

    def atualizar_parcial(self, hipotese_palavras, hipotese_anterior):
        comum = _prefixo_comum(hipotese_anterior, hipotese_palavras)
        if comum > self.confirmadas:
            self.confirmadas = comum
            self.palavras = hipotese_palavras[:comum]
        # mostra confirmado + o "rascunho" ainda instavel do fim, pra dar
        # feedback visual mesmo antes de travar as palavras
        rascunho = " ".join(hipotese_palavras)
        sys.stdout.write(f"\r[legenda...] {rascunho}" + " " * 15)
        sys.stdout.flush()

    def finalizar(self, texto_final, latencia):
        sys.stdout.write(
            f"\r[legenda]    {texto_final}   (final, {latencia:.2f}s)" + " " * 15 + "\n"
        )
        sys.stdout.flush()


def processar_fala():
    print(f"[modelo] carregando faster-whisper '{MODELO_WHISPER}' ({DEVICE}/{COMPUTE_TYPE})...", flush=True)
    try:
        modelo = WhisperModel(MODELO_WHISPER, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        # Maquina sem GPU/CUDA (ex: notebook de outro integrante do time) -> cai pra CPU
        # em vez de quebrar o programa.
        print(f"[modelo] falhou em '{DEVICE}' ({e}). Caindo para cpu/int8...", flush=True)
        modelo = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")
    print("[modelo] pronto. Fale ou toque audio na call para ver a legenda.\n", flush=True)

    vad = webrtcvad.Vad(VAD_AGRESSIVIDADE)
    ring_inicio = collections.deque(maxlen=FRAMES_JANELA)
    ring_fim = collections.deque(maxlen=FRAMES_JANELA_FIM)

    em_fala = False
    segmento = []
    inicio_segmento = None
    ultimo_parcial = 0.0
    hipotese_anterior = []
    legenda = None

    def finalizar_segmento():
        nonlocal segmento, hipotese_anterior, legenda
        if not segmento:
            return
        audio_np = _pcm_para_float(segmento)
        if len(audio_np) >= SAMPLE_RATE * 0.3:  # ignora ruidos curtos (<300ms)
            t0 = time.time()
            texto = _transcrever(modelo, audio_np)
            latencia = time.time() - t0
            if texto:
                if legenda is None:
                    legenda = LegendaAoVivo()
                legenda.finalizar(texto, latencia)
        segmento = []
        hipotese_anterior = []
        legenda = None

    while True:
        frame = fila_frames.get()
        if frame is None:
            break

        voz = vad.is_speech(frame, SAMPLE_RATE)

        if not em_fala:
            ring_inicio.append((frame, voz))
            num_voz = len([f for f, v in ring_inicio if v])
            if num_voz > RATIO_INICIO_FALA * ring_inicio.maxlen:
                em_fala = True
                inicio_segmento = time.time()
                ultimo_parcial = time.time()
                segmento.extend(f for f, _ in ring_inicio)
                ring_inicio.clear()
                ring_fim.clear()
        else:
            segmento.append(frame)
            ring_fim.append((frame, voz))
            num_silencio = len([f for f, v in ring_fim if not v])
            duracao = time.time() - inicio_segmento
            janela_cheia = len(ring_fim) == ring_fim.maxlen
            fim_por_pausa = janela_cheia and num_silencio > RATIO_FIM_FALA * ring_fim.maxlen

            if fim_por_pausa or duracao > MAX_SEGMENTO_S:
                finalizar_segmento()
                ring_inicio.clear()
                ring_fim.clear()
                em_fala = False
                continue

            # ainda falando: de tempos em tempos, refaz a transcricao do que
            # ja foi dito ate agora e confirma as palavras estaveis - isso e'
            # o que faz a legenda aparecer ENQUANTO a pessoa fala, em vez de
            # so no final da frase.
            audio_acumulado_s = len(segmento) * FRAME_MS / 1000
            if (
                time.time() - ultimo_parcial >= PARTIAL_INTERVAL_S
                and audio_acumulado_s >= MIN_AUDIO_PARCIAL_S
            ):
                audio_np = _pcm_para_float(segmento)
                texto = _transcrever(modelo, audio_np)
                palavras = texto.split()
                if legenda is None:
                    legenda = LegendaAoVivo()
                legenda.atualizar_parcial(palavras, hipotese_anterior)
                hipotese_anterior = palavras
                ultimo_parcial = time.time()

    finalizar_segmento()


def main():
    threads = [
        threading.Thread(target=capturar_audio, daemon=True),
        threading.Thread(target=processar_fala, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[main] encerrando...", flush=True)
        parar.set()
        for t in threads:
            t.join(timeout=2)


if __name__ == "__main__":
    main()
