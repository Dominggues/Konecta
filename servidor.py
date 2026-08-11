"""
Servico central local ("interface central" que o Vini mencionou).

Roda 100% na sua maquina (nao precisa de internet nem de conta em nenhum
servico externo). Qualquer modulo do time publica texto aqui via HTTP, e
qualquer outro modulo (a legenda flutuante, e futuramente o avatar em
Libras do Vinicius) escuta essas mensagens em tempo real via WebSocket.

Contrato de mensagem (o formato que cada modulo deve publicar/receber):

    {
        "origem": "audio" | "libras",   # audio = veio do transcricao_tempo_real.py
                                         # libras = veio do reconhecimento de sinais (Vinicius)
        "tipo": "parcial" | "final",    # final = frase fechada; parcial = ainda em andamento
        "texto": "...",
        "latencia_s": 0.35              # opcional, so' faz sentido em "final"
    }

Como rodar:

    .venv\\Scripts\\python.exe servidor.py

Fica escutando em http://127.0.0.1:8765
  - POST /publicar   -> qualquer modulo manda texto novo pra ca'
  - GET  /health      -> checagem simples (util pros outros modulos saberem
                          se o servidor esta' de pe' antes de tentar publicar)
  - WS   /ws           -> quem quiser RECEBER as mensagens em tempo real
                          conecta aqui (e' o que a legenda_flutuante.py faz)
  - GET  /avatar        -> pagina de demo com o widget oficial do VLibras
                          (essa parte especifica PRECISA de internet, e' a
                          unica excecao ao "100% local" - o resto do servico
                          continua rodando sem rede nenhuma)
"""

import json
import os
from typing import List, Literal, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

PORTA = 8765
CAMINHO_AVATAR_HTML = os.path.join(os.path.dirname(__file__), "avatar_vlibras_demo.html")

app = FastAPI(title="Konecta - servico central")


@app.get("/avatar")
def avatar():
    if not os.path.isfile(CAMINHO_AVATAR_HTML):
        return PlainTextResponse("avatar_vlibras_demo.html nao encontrado.", status_code=404)
    return FileResponse(CAMINHO_AVATAR_HTML, media_type="text/html")


class Mensagem(BaseModel):
    origem: Literal["audio", "libras"]
    tipo: Literal["parcial", "final"]
    texto: str
    latencia_s: Optional[float] = None


class GerenciadorDeConexoes:
    """Mantem a lista de quem esta' conectado no WebSocket e distribui as
    mensagens publicadas para todo mundo (broadcast)."""

    def __init__(self):
        self.conexoes: List[WebSocket] = []

    async def conectar(self, ws: WebSocket):
        await ws.accept()
        self.conexoes.append(ws)
        print(f"[servidor] cliente conectado ({len(self.conexoes)} no total)")

    def desconectar(self, ws: WebSocket):
        if ws in self.conexoes:
            self.conexoes.remove(ws)
        print(f"[servidor] cliente desconectado ({len(self.conexoes)} no total)")

    async def transmitir(self, mensagem: dict):
        mortas = []
        for ws in self.conexoes:
            try:
                await ws.send_text(json.dumps(mensagem, ensure_ascii=False))
            except Exception:
                mortas.append(ws)
        for ws in mortas:
            self.desconectar(ws)


gerenciador = GerenciadorDeConexoes()


@app.get("/health")
def health():
    return {"status": "ok", "clientes_conectados": len(gerenciador.conexoes)}


@app.post("/publicar")
async def publicar(msg: Mensagem):
    print(f"[servidor] [{msg.origem}/{msg.tipo}] {msg.texto}")
    await gerenciador.transmitir(msg.model_dump())
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await gerenciador.conectar(ws)
    try:
        while True:
            # nao esperamos nada do cliente, so' mantemos a conexao viva
            await ws.receive_text()
    except WebSocketDisconnect:
        gerenciador.desconectar(ws)


if __name__ == "__main__":
    print(f"[servidor] subindo em http://127.0.0.1:{PORTA}  (Ctrl+C p/ parar)")
    uvicorn.run(app, host="127.0.0.1", port=PORTA, log_level="warning")
