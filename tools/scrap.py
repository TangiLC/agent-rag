from pathlib import Path
from weasyprint import HTML, CSS

URL_TO_PDF = [
    {"url": "https://fr.wikipedia.org/wiki/Faucon_p%C3%A8lerin", "file_name": "faucon_pelerin.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Aigle_royal", "file_name": "aigle_royal.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Albatros_%C3%A0_t%C3%AAte_grise", "file_name": "albatros_tete_grise.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Faucon_gerfaut", "file_name": "faucon_gerfaut.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Martinet_noir", "file_name": "martinet_noir.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Fr%C3%A9gate_ariel", "file_name": "fregate_ariel.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Oie-arm%C3%A9e_de_Gambie", "file_name": "oie_gambie.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Harle_hupp%C3%A9", "file_name": "harle_huppe.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Fuligule_%C3%A0_dos_blanc", "file_name": "fuligule_dos_blanc.pdf"},
    {"url": "https://fr.wikipedia.org/wiki/Go%C3%A9land_brun", "file_name": "goeland_brun.pdf"},
]

OUT_DIR = Path("rag_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CSS de filtrage
WIKI_CLEAN_CSS = CSS(string="""
    /* Supprimer toutes les images */
    img,
    figure,
    .infobox,
    .thumb,
    .thumbinner {
        display: none !important;
    }

    /* Supprimer références, notes, bibliographie */
    .reference,
    .references,
    #references,
    .mw-references-wrap,
    ol.references,
    sup.reference {
        display: none !important;
    }

    /* Supprimer sections inutiles */
    #Notes,
    #Références,
    #Bibliographie,
    #Voir_aussi,
    #Liens_externes {
        display: none !important;
    }
""")

def convert_urls_to_pdfs(items, out_dir: Path) -> None:
    for i, item in enumerate(items, start=1):
        url = item["url"]
        file_name = item["file_name"]
        out_path = out_dir / file_name

        print(f"[{i}/{len(items)}] {url} -> {out_path}")

        try:
            HTML(url).write_pdf(
                str(out_path),
                stylesheets=[WIKI_CLEAN_CSS],
                presentational_hints=False
            )
        except Exception as e:
            print(f"  !! ERREUR ({type(e).__name__}): {e}")

def main() -> None:
    if not URL_TO_PDF:
        print("URL_TO_PDF est vide.")
        return

    convert_urls_to_pdfs(URL_TO_PDF, OUT_DIR)
    print("Terminé.")

if __name__ == "__main__":
    main()
