import requests
import pypdf
import io
import os

pdf_url = "https://www.marche.camcom.it/tutela-impresa-e-consumatore/albi-e-ruoli/elenco-raccomandatari-marittimi/codice_navigazione.pdf"
data_dir = "data"
output_text_filename = os.path.join(data_dir, "extracted_codice_navigazione.txt")

def extract_text_from_pdf_url(url):
    print(f"Downloading PDF from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        pdf_file = io.BytesIO(response.content)

        reader = pypdf.PdfReader(pdf_file)

        extracted_text = ""
        print(f"Total pages numbers: {len(reader.pages)}")
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text += page.extract_text() + "\n--- Pagina Fine ---\n"

        print("Text extraction completed.")
        return extracted_text

    except requests.exceptions.RequestException as e:
        print(f"Scraping error {url}: {e}")
        return None
    except Exception as e:
        print(f"Analytics error {url}: {e}")
        return None

if __name__ == "__main__":
    pdf_text = extract_text_from_pdf_url(pdf_url)

    os.makedirs(data_dir, exist_ok=True)

    if pdf_text:
        with open(output_text_filename, 'w', encoding='utf-8') as f:
            f.write(pdf_text)

        print(f"Text saved in -> '{output_text_filename}'")

