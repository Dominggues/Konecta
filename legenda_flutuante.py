"""
Janela flutuante que fica sempre por cima de qualquer programa (inclusive
por cima do Teams/Meet durante a call), mostrando a legenda em tempo real.

Conecta no servidor.py via WebSocket e so' escuta - nao faz nenhum
processamento de audio/video, so' exibe o que os outros modulos publicam.

Como rodar (nessa ordem):

    1. .venv\\Scripts\\python.exe servidor.py
    2. .venv\\Scripts\\python.exe legenda_flutuante.py
    3. .venv\\Scripts\\python.exe transcricao_tempo_real.py

Arraste a janela pra onde quiser na tela. Fecha com o "X" ou Alt+F4.
"""

import json
import threading
import tkinter as tk

import websocket  # pacote 'websocket-client'

URL_WS = "ws://127.0.0.1:8765/ws"

COR_FUNDO = "#101010"
COR_TEXTO_AUDIO = "#ffffff"
COR_TEXTO_LIBRAS = "#7ec8ff"
MAX_LINHAS = 4


class LegendaFlutuante:
    def __init__(self):
        self.janela = tk.Tk()
        self.janela.title("Konecta - Legenda")
        self.janela.attributes("-topmost", True)   # sempre por cima
        self.janela.attributes("-alpha", 0.88)      # levemente transparente
        self.janela.overrideredirect(True)          # sem borda/barra de titulo padrao
        self.janela.configure(bg=COR_FUNDO)

        largura, altura = 900, 220
        tela_w = self.janela.winfo_screenwidth()
        x = (tela_w - largura) // 2
        y = self.janela.winfo_screenheight() - altura - 80  # perto do rodape, como legenda de call
        self.janela.geometry(f"{largura}x{altura}+{x}+{y}")

        self._montar_barra_de_arrastar()

        # Text em vez de Label: Label corta o conteudo que nao cabe na altura
        # da janela (foi o que causou o bug visual reportado - frase comprida
        # quebrando em varias linhas e a de cima ficando cortada). Text tem
        # scroll de verdade: sempre mostra o final, o que passou some suave.
        self.texto_widget = tk.Text(
            self.janela,
            font=("Segoe UI", 16),
            bg=COR_FUNDO,
            fg=COR_TEXTO_AUDIO,
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=20,
            pady=10,
            state="disabled",
            cursor="arrow",
        )
        self.texto_widget.pack(expand=True, fill="both")
        self.texto_widget.tag_configure("audio", foreground=COR_TEXTO_AUDIO)
        self.texto_widget.tag_configure("libras", foreground=COR_TEXTO_LIBRAS)
        self.texto_widget.tag_configure("status", foreground="#888888")

        self.linhas = []  # historico curto de (origem, texto) ja finalizados
        self.texto_parcial = ""
        self.origem_parcial = "audio"

        self._set_status("Aguardando o servidor e a transcricao...")

        self._iniciar_cliente_ws()

    def _montar_barra_de_arrastar(self):
        """Como a janela nao tem borda, precisamos de uma faixa no topo pra
        poder arrasta-la e um botao de fechar."""
        barra = tk.Frame(self.janela, bg="#202020", height=24)
        barra.pack(side="top", fill="x")

        tk.Label(
            barra, text="⠿ arraste aqui  •  Konecta", bg="#202020", fg="#888888",
            font=("Segoe UI", 8),
        ).pack(side="left", padx=8)

        tk.Button(
            barra, text="x", bg="#202020", fg="#888888", bd=0,
            command=self.janela.destroy, font=("Segoe UI", 9),
        ).pack(side="right", padx=4)

        def iniciar_arrasto(evento):
            self._offset_x = evento.x
            self._offset_y = evento.y

        def arrastar(evento):
            x = self.janela.winfo_x() + evento.x - self._offset_x
            y = self.janela.winfo_y() + evento.y - self._offset_y
            self.janela.geometry(f"+{x}+{y}")

        barra.bind("<Button-1>", iniciar_arrasto)
        barra.bind("<B1-Motion>", arrastar)

    def _iniciar_cliente_ws(self):
        thread = threading.Thread(target=self._loop_ws, daemon=True)
        thread.start()

    def _loop_ws(self):
        def ao_receber(ws, mensagem_bruta):
            try:
                msg = json.loads(mensagem_bruta)
            except json.JSONDecodeError:
                return
            # atualiza a UI a partir da thread principal do tkinter
            self.janela.after(0, self._processar_mensagem, msg)

        def ao_conectar(ws):
            self.janela.after(0, self._set_status, "Conectado. Aguardando fala...")

        def ao_erro(ws, erro):
            self.janela.after(0, self._set_status, f"Servidor indisponivel ({erro})")

        def ao_fechar(ws, *_args):
            self.janela.after(0, self._set_status, "Desconectado do servidor.")

        while True:
            try:
                app_ws = websocket.WebSocketApp(
                    URL_WS,
                    on_open=ao_conectar,
                    on_message=ao_receber,
                    on_error=ao_erro,
                    on_close=ao_fechar,
                )
                app_ws.run_forever(reconnect=3)
            except Exception:
                pass

    def _set_status(self, texto):
        self.texto_widget.config(state="normal")
        self.texto_widget.delete("1.0", "end")
        self.texto_widget.insert("end", texto, "status")
        self.texto_widget.config(state="disabled")

    def _processar_mensagem(self, msg):
        origem = msg.get("origem", "audio")
        tipo = msg.get("tipo")
        texto = msg.get("texto", "")

        if tipo == "parcial":
            self.texto_parcial = texto
            self.origem_parcial = origem
        elif tipo == "final":
            self.linhas.append((origem, texto))
            self.linhas = self.linhas[-MAX_LINHAS:]
            self.texto_parcial = ""

        self._renderizar()

    def _renderizar(self):
        self.texto_widget.config(state="normal")
        self.texto_widget.delete("1.0", "end")
        for origem, texto in self.linhas:
            tag = "libras" if origem == "libras" else "audio"
            self.texto_widget.insert("end", texto + "\n", tag)
        if self.texto_parcial:
            tag = "libras" if self.origem_parcial == "libras" else "audio"
            self.texto_widget.insert("end", self.texto_parcial, tag)
        self.texto_widget.see("end")  # rola pro final - e' isso que resolve o corte
        self.texto_widget.config(state="disabled")

    def executar(self):
        self.janela.mainloop()


if __name__ == "__main__":
    LegendaFlutuante().executar()
