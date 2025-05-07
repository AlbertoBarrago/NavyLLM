import requests
import pypdf
import io
import json


pdf_url = "https://www.marche.camcom.it/tutela-impresa-e-consumatore/albi-e-ruoli/elenco-raccomandatari-marittimi/codice_navigazione.pdf"
OUTPUT_JSON_FILENAME = "../extracted_codice_navigazione_for_rag.json"

def extract_text_from_pdf_url(url):
    """Scarica un PDF da un URL ed estrae il testo."""
    print(f"Scaricando il PDF da: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        pdf_file = io.BytesIO(response.content)

        reader = pypdf.PdfReader(pdf_file)

        extracted_text = ""
        print(f"Numero totale di pagine: {len(reader.pages)}")
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text += page.extract_text() + "\n--- Pagina Fine ---\n"

        print("Estrazione testo completata.")
        return extracted_text

    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download del PDF da {url}: {e}")
        return None
    except Exception as e:
        print(f"Errore durante l'analisi di {url}: {e}")
        return None

# --- Esecuzione dello script ---
if __name__ == "__main__":
    pdf_text = extract_text_from_pdf_url(pdf_url)

    if pdf_text:
        data_for_rag_index = [
            {
                "url": pdf_url,
                "text": pdf_text
            }
        ]

        with open(OUTPUT_JSON_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(data_for_rag_index, f, ensure_ascii=False, indent=4)

        print(f"Testo estratto dal PDF e salvato in formato JSON in '{OUTPUT_JSON_FILENAME}'")

        print(f"\nOra puoi usare '{OUTPUT_JSON_FILENAME}' come input per lo script build_rag_index.py.")
        print(f"Modifica la variabile RAW_TEXT_DATA_FILE in build_rag_index.py per puntare a questo file.")

