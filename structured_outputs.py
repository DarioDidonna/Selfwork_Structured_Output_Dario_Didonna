import os
import json
import requests
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Carica le variabili d'ambiente (.env)
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def get_free_model() -> str:
    """Seleziona dinamicamente un modello gratuito attivo su OpenRouter."""
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            # Cerca primariamente un modello Gemini gratuito
            for model in models:
                model_id = model.get("id", "")
                if "gemini" in model_id and model_id.endswith(":free"):
                    print(f"Modello Gemini selezionato: {model_id}")
                    return model_id
            # Fallback su qualsiasi modello :free
            for model in models:
                model_id = model.get("id", "")
                if model_id.endswith(":free"):
                    print(f"Modello gratuito selezionato: {model_id}")
                    return model_id
    except Exception as e:
        print(f"Errore nel recupero modelli: {e}")
    
    return "google/gemini-2.0-flash-exp:free"

MODEL_NAME = get_free_model()


# =====================================================================
# ESEMPIO 1: Output Strutturato con Google Gemini (Estrazione Dati)
# =====================================================================

class SchedaProdotto(BaseModel):
    nome_prodotto: str = Field(description="Nome commerciale del prodotto")
    categoria: str = Field(description="Categoria di appartenenza (es. Elettronica, Abbigliamento)")
    prezzo_stimato: float = Field(description="Prezzo espresso in numeri decimali")
    caratteristiche_principali: List[str] = Field(description="Lista delle funzionalità chiave")
    valutazione_pro: List[str] = Field(description="Punti di forza del prodotto")
    valutazione_contro: List[str] = Field(description="Eventuali difetti o limitazioni")


def esempio_structured_output_gemini():
    print("\n" + "="*60)
    print("ESEMPIO 1: Output Strutturato Pydantic con Gemini")
    print("="*60)

    recensione_testo = """
    Abbiamo provato il nuovo laptop TechBook Pro 15. Si inserisce nella categoria dell'informatica per professionisti.
    Viene venduto a circa 1299.99 euro. Offre uno schermo OLED eccezionale, processore a 12 core e 32GB di RAM.
    Tra i punti di forza troviamo l'autonomia della batteria oltre le 14 ore e la silenziosità delle ventole.
    Di contro, la tastiera ha una corsa un po' breve e manca la porta Ethernet integrata.
    """

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Sei un estrattore di dati preciso. "
                    "Rispondi ESCLUSIVAMENTE con un JSON valido conforme a questo schema Pydantic:\n"
                    f"{json.dumps(SchedaProdotto.model_json_schema())}"
                )
            },
            {"role": "user", "content": recensione_testo}
        ],
        response_format={"type": "json_object"},
    )

    raw_json = completion.choices[0].message.content
    prodotto = SchedaProdotto.model_validate_json(raw_json)

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

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Sei un assistente editoriale specializzato nella sintesi di articoli complessi. "
                    "Analizza il testo fornito e restituisci la sintesi in un formato JSON rigorosamente conforme a questo schema:\n"
                    f"{json.dumps(SintesiArticolo.model_json_schema())}"
                )
            },
            {"role": "user", "content": articolo}
        ],
        response_format={"type": "json_object"},
    )

    raw_json = completion.choices[0].message.content
    sintesi = SintesiArticolo.model_validate_json(raw_json)

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
    esempio_structured_output_gemini()
    esempio_text_summarization()