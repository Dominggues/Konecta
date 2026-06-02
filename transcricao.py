from faster_whisper import WhisperModel
import time

# Configurações do modelo
TAMANHO_MODELO = "base" 
ARQUIVO_AUDIO = "teste_reuniao.wav"

def transcrever_audio():
    print("⏳ Carregando o modelo de IA (Isso pode demorar um pouquinho na primeira vez para baixar os arquivos)...")
    
    # Inicia o modelo. Se você tiver uma placa de vídeo da NVIDIA, mude device="cpu" para device="cuda"
    modelo = WhisperModel(TAMANHO_MODELO, device="cpu", compute_type="int8")
    
    print(f"🎙️ Transcrevendo o arquivo: {ARQUIVO_AUDIO}")
    
    # Marca o tempo de início para vermos a performance
    tempo_inicio = time.time()
    
    # Realiza a transcrição
    # language="pt" força a IA a focar no português, acelerando o processo
    segmentos, informacoes = modelo.transcribe(ARQUIVO_AUDIO, language="pt", beam_size=5)
    
    print(f"\n✅ Idioma detectado: '{informacoes.language}' com {informacoes.language_probability * 100:.2f}% de certeza.\n")
    print("📝 --- TEXTO TRADUZIDO ---")
    
    texto_completo = ""
    for segmento in segmentos:
        print(f"[{segmento.start:.2f}s -> {segmento.end:.2f}s] {segmento.text}")
        texto_completo += segmento.text + " "
        
    tempo_fim = time.time()
    tempo_total = tempo_fim - tempo_inicio
    
    print("\n-------------------------")
    print(f"⏱️ Tempo de processamento: {tempo_total:.2f} segundos")
    
if __name__ == "__main__":
    transcrever_audio()