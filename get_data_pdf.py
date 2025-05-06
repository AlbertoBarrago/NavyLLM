import requests
import pypdf
import io

pdf_url = "https://www.marche.camcom.it/tutela-impresa-e-consumatore/albi-e-ruoli/elenco-raccomandatari-marittimi/codice_navigazione.pdf"

output_text_filename = "extracted_codice_navigazione_text.txt"

def extract_text_from_pdf_url(url):
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
        print(f"Errore durante lo scraping di {url}: {e}")
        return None
    except Exception as e:
        print(f"Errore durante l'analisi di {url}: {e}")
        return None

if __name__ == "__main__":
    pdf_text = extract_text_from_pdf_url(pdf_url)

    if pdf_text:
        with open(output_text_filename, 'w', encoding='utf-8') as f:
            f.write(pdf_text)

        print(f"Testo estratto salvato in '{output_text_filename}'")

