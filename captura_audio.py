import soundcard as sc
import soundfile as sf

# Configurações do Áudio
NOME_ARQUIVO = "teste_reuniao.wav"
TAXA_AMOSTRAGEM = 44100  # Frequência padrão (44.1kHz)
TEMPO_GRAVACAO = 15      # Tempo em segundos da gravação

def hello_world_audio():
    # 1. Busca o alto-falante padrão do Windows
    alto_falante_padrao = sc.default_speaker()
    print(f"🎤 Buscando loopback do dispositivo: {alto_falante_padrao.name}")
    
    # 2. Localiza o "microfone interno" (loopback) que escuta esse alto-falante
    mic_loopback = sc.get_microphone(id=str(alto_falante_padrao.name), include_loopback=True)
    
    print(f"\n🎧 Preparando para gravar {TEMPO_GRAVACAO} segundos.")
    print("▶️ COLOQUE ALGUM VÍDEO NO YOUTUBE OU ÁUDIO PARA TOCAR AGORA!")
    
    # 3. Abre o microfone virtual e grava
    with mic_loopback.recorder(samplerate=TAXA_AMOSTRAGEM) as mic:
        print("🔴 Gravando...")
        
        # Pega os frames de áudio multiplicando a taxa pelo tempo desejado
        dados_audio = mic.record(numframes=TAXA_AMOSTRAGEM * TEMPO_GRAVACAO)
        
        print("⏹️ Gravação concluída!")
        
        # 4. Salva o áudio no formato WAV para testarmos depois com a IA de texto
        sf.write(file=NOME_ARQUIVO, data=dados_audio, samplerate=TAXA_AMOSTRAGEM)
        print(f"💾 Arquivo salvo com sucesso: {NOME_ARQUIVO}")

if __name__ == "__main__":
    hello_world_audio()