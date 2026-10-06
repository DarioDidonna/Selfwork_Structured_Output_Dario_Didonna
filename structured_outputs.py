import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Carica le variabili d'ambiente (.env)
load_dotenv()

# Inizializza il client OpenAI nativo
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"


# =====================================================================
# ESEMPIO 1: Output Strutturato con OpenAI (Estrazione Dati)
# =====================================================================

class SchedaProdotto(BaseModel):
    nome_prodotto: str = Field(description="Nome commerciale del prodotto")
    categoria: str = Field(description="Categoria di appartenenza (es. Elettronica, Abbigliamento)")
    prezzo_stimato: float = Field(description="Prezzo espresso in numeri decimali")
    caratteristiche_principali: List[str] = Field(description="Lista delle funzionalità chiave")
    valutazione_pro: List[str] = Field(description="Punti di forza del prodotto")
    valutazione_contro: List[str] = Field(description="Eventuali difetti o limitazioni")


def esempio_structured_output_openai():
    print("\n" + "="*60)
    print("ESEMPIO 1: Output Strutturato Pydantic con OpenAI")
    print("="*60)

    recensione_testo = """
    Abbiamo provato il nuovo laptop TechBook Pro 15. Si inserisce nella categoria dell'informatica per professionisti.
    Viene venduto a circa 1299.99 euro. Offre uno schermo OLED eccezionale, processore a 12 core e 32GB di RAM.
    Tra i punti di forza troviamo l'autonomia della batteria oltre le 14 ore e la silenziosità delle ventole.
    Di contro, la tastiera ha una corsa un po' breve e manca la porta Ethernet integrata.
    """

    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Sei un estrattore di dati preciso."},
            {"role": "user", "content": recensione_testo}
        ],
        response_format=SchedaProdotto,
    )

    prodotto = completion.choices[0].message.parsed

    print(f"💻 Prodotto: {prodotto.nome_prodotto}")
    print(f"📁 Categoria: {prodotto.categoria}")
    print(f"💰 Prezzo: €{prodotto.prezzo_stimato:.2f}")
    print(f"⚡ Features: {', '.join(prodotto.caratteristiche_principali)}")
    print(f"✅ Pro: {', '.join(prodotto.valutazione_pro)}")
    print(f"❌ Contro: {', '.join(prodotto.valutazione_contro)}")


# =====================================================================
# ESEMPIO 2: Text Summarization Strutturata (Sintesi di un Articolo)
# =====================================================================

class SintesiArticolo(BaseModel):
    titolo_sintetico: str = Field(description="Un titolo incisivo e riassuntivo per l'articolo")
    argomento_principale: str = Field(description="In una frase, di cosa parla l'articolo")
    punti_chiave: List[str] = Field(description="Elenco di 3-5 punti fondamentali emersi nel testo")
    tempo_lettura_stimato_minuti: int = Field(description="Stima del tempo necessario per leggere l'originale")
    conclusione: str = Field(description="La conclusione o presa di posizione finale del testo")


def esempio_text_summarization():
    print("\n" + "="*60)
    print("ESEMPIO 2: Text Summarization Strutturata")
    print("="*60)

    articolo = """
    L'adozione dell'Intelligenza Artificiale nelle piccole e medie imprese italiane ha registrato un incremento del 35%
    nell'ultimo anno. Secondo il report di settore, le aziende utilizzano l'AI principalmente per l'automazione del servizio
    clienti tramite chatbot e per l'analisi predittiva delle vendite. Tuttavia, emergono due principali ostacoli: la mancanza
    di competenze interne specializzate e la preoccupazione per i costi di integrazione iniziale. Per superare queste sfide,
    molte PMI stanno optando per soluzioni SaaS pronte all'uso e per la formazione del personale esistente. Gli esperti
    concludono che nei prossimi tre anni l'adozione dell'AI diventerà un requisito fondamentale di competitività, non più un opzionale.
    """

    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Sei un assistente editoriale specializzato nella sintesi di articoli complessi."},
            {"role": "user", "content": articolo}
        ],
        response_format=SintesiArticolo,
    )

    sintesi = completion.choices[0].message.parsed

    print(f"📰 Titolo: {sintesi.titolo_sintetico}")
    print(f"🎯 Argomento: {sintesi.argomento_principale}")
    print(f"⏱️ Tempo lettura testo originale: ~{sintesi.tempo_lettura_stimato_minuti} min")
    print("\n📌 Punti Chiave:")
    for punto in sintesi.punti_chiave:
        print(f"  • {punto}")
    print(f"\n💡 Conclusione: {sintesi.conclusione}")


# =====================================================================
# ESECUZIONE
# =====================================================================

if __name__ == "__main__":
    esempio_structured_output_openai()
    esempio_text_summarization()
