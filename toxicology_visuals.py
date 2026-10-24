"""Imágenes de referencia para fichas de tóxicos externos MEDCALC.

Fuentes: Wikimedia Commons (fotos reutilizables con créditos y licencia).
Sin fotografía suficientemente identificada, mostrar una ilustración original
esquemática, nunca una foto no verificada de otra especie o sustancia.
"""
from __future__ import annotations

import html
import re
import unicodedata
from xml.sax.saxutils import escape

import requests
import streamlit as st

API = "https://commons.wikimedia.org/w/api.php"
LICENSES = ("cc0", "cc-by", "cc by", "public domain", "pd-", "pdm")

# Términos taxonómicos específicos. En entradas compuestas se muestran dos fotos.
VISUAL_QUERIES = {
    "LOXOSCELES": ("Loxosceles laeta",),
    "LATRODECTUS": ("Latrodectus mactans",),
    "PHONEUTRIA": ("Phoneutria nigriventer",),
    "ATRAX": ("Atrax robustus",),
    "TARANTUL": ("Theraphosidae tarantula",),
    "TITYUS": ("Tityus serrulatus",),
    "CENTRUROIDES": ("Centruroides sculpturatus",),
    "BUTHIDAE": ("Leiurus quinquestriatus",),
    "BOTHROPS": ("Bothrops jararaca",),
    "CROTALUS": ("Crotalus durissus",),
    "LACHESIS": ("Lachesis muta",),
    "MICRURUS": ("Micrurus coral snake",),
    "COBRAS": ("Naja naja",),
    "KRAITS": ("Bungarus fasciatus",),
    "MAMBAS": ("Dendroaspis polylepis",),
    "SERPIENTES MARINAS": ("Hydrophis platurus",),
    "MORDEDURA DE SERPIENTE": ("Bothrops snake",),
    "MONSTRUO DE GILA": ("Heloderma suspectum",),
    "ABEJAS": ("Apis mellifera",),
    "AVISPAS": ("Vespula germanica",),
    "HORMIGA DE FUEGO": ("Solenopsis invicta",),
    "LONOMIA": ("Lonomia obliqua",),
    "MEGALOPYGE": ("Megalopyge opercularis",),
    "GARRAPATA": ("Ixodes ricinus",),
    "FRAGATA PORTUGUESA": ("Physalia physalis",),
    "CHIRONEX": ("Chironex fleckeri",),
    "IRUKANDJI": ("Carukia barnesi",),
    "PEZ PIEDRA": ("Synanceia verrucosa",),
    "PEZ LEON": ("Pterois volitans",),
    "RAYA /": ("Dasyatis pastinaca",),
    "CARACOL CONO": ("Conus geographus",),
    "PULPO DE ANILLOS": ("Hapalochlaena lunulata",),
    "ERIZO DE MAR": ("Paracentrotus lividus",),
    "ESTRELLA CORONA": ("Acanthaster planci",),
    "ANEMONA": ("Actinia equina",),
    "PEZ GATO": ("Plotosus lineatus",),
    "WEEVERFISH": ("Trachinus draco",),
    "PEZ ARANA": ("Trachinus draco",),
    "PEZ GLOBO": ("Takifugu rubripes",),
    "TETRODOTOXINA": ("Takifugu rubripes",),
    "ZOANTIDOS": ("Zoanthus pulchellus",),
    "PALITOXINA": ("Palythoa tuberculosa",),
    "CIGUATERA": ("Sphyraena barracuda",),
    "ESCOMBROIDE": ("Scomber scombrus",),
    "MARISCOS": ("Mytilus edulis",),
    "BUFOTOXINAS": ("Rhinella marina",),
    "ADELFA": ("Nerium oleander",),
    "DIGITAL /": ("Digitalis purpurea",),
    "DATURA": ("Datura stramonium", "Brugmansia suaveolens"),
    "TEJO": ("Taxus baccata",),
    "RICINO": ("Ricinus communis",),
    "ACONITO": ("Aconitum napellus",),
    "CICUTA": ("Conium maculatum",),
    "COLCHICUM": ("Colchicum autumnale",),
    "AMATOXINAS": ("Amanita phalloides",),
    "GYROMITRINA": ("Gyromitra esculenta",),
    "MUSCARINICO POR HONGOS": ("Inocybe erubescens",),
    "AMANITA MUSCARIA": ("Amanita muscaria",),
    "ORELLANINA": ("Cortinarius orellanus",),
    "COPRINA": ("Coprinopsis atramentaria",),
    "PSILOCIBINA": ("Psilocybe cubensis",),
    "HONGOS IRRITANTES": ("Chlorophyllum molybdites",),
    "ACEBO / MUERDAGO": ("Ilex aquifolium", "Viscum album"),
    "BRIONIA": ("Bryonia dioica",),
    "VERATRO": ("Veratrum album",),
    "LIRIO DE LOS VALLES": ("Convallaria majalis",),
    "CASSAVA": ("Manihot esculenta",),
    "ESCOLOPENDRA": ("Scolopendra subspinipes",),
    "PROCESIONARIA": ("Thaumetopoea pityocampa",),
    "VIPERA": ("Vipera aspis",),
    "MEDUSAS": ("Aurelia aurita",),
    "MEDUSA /": ("Aurelia aurita",),
}
BIO_CATEGORIES = {
    "ARACNIDOS", "ESCORPIONES", "SERPIENTES", "REPTILES",
    "HIMENOPTEROS", "ORUGAS", "ARTROPODOS", "MARINOS",
    "TOXINAS MARINAS", "ANIMALES", "PLANTAS", "HONGOS",
}


def _norm(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value.upper()).strip()


def _queries(row):
    name = _norm(row.get("toxico"))
    category = _norm(row.get("categoria"))
    for marker, queries in VISUAL_QUERIES.items():
        if marker in name:
            return queries
    # La fotografía de animales/plantas solo se busca para un taxón definido.
    if category in BIO_CATEGORIES:
        scientific = re.search(r"\b[A-Z][A-Z-]{4,}\s+(?:[A-Z][A-Z-]{3,})\b", name)
        if scientific:
            return (scientific.group(0).title(),)
    return ()


def _plain_html(text):
    return html.unescape(re.sub(r"<[^>]*>", "", str(text or ""))).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def _commons_photo(query):
    """Una imagen Commons con licencia permisiva y título que mencione el taxón."""
    try:
        response = requests.get(
            API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f'filetype:bitmap "{query}"',
                "gsrnamespace": 6,
                "gsrlimit": 12,
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit",
                "iiurlwidth": 700,
                "format": "json",
                "formatversion": 2,
            },
            timeout=7,
            headers={"User-Agent": "MedCalc-Clinico/1.0 (medical-reference; Wikimedia Commons attribution)"},
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)]
        for page in sorted(pages, key=lambda x: x.get("index", 999)):
            title = _norm(page.get("title"))
            # Evita mostrar otra especie porque el texto buscado aparezca solo en la descripción.
            if not words or not all(w.upper() in title for w in words[:2]):
                continue
            info = (page.get("imageinfo") or [{}])[0]
            mime = str(info.get("mime") or "")
            if mime not in ("image/jpeg", "image/png", "image/webp"):
                continue
            metadata = info.get("extmetadata") or {}
            license_name = _plain_html((metadata.get("LicenseShortName") or {}).get("value", ""))
            if not any(k in license_name.lower() for k in LICENSES):
                continue
            url = info.get("thumburl") or info.get("url")
            source = info.get("descriptionurl")
            if not str(url).startswith("https://") or not str(source).startswith("https://"):
                continue
            return {
                "image_url": url,
                "page_url": source,
                "license": license_name,
                "license_url": (metadata.get("LicenseUrl") or {}).get("value", ""),
                "author": _plain_html((metadata.get("Artist") or {}).get("value", ""))[:180],
                "scientific": query,
            }
    except (requests.RequestException, ValueError, TypeError, KeyError):
        pass
    return None


def _svg_reference(name, category):
    """Ilustración propia de referencia cuando no hay foto licenciada verificada."""
    safe_name = escape(str(name or "Tóxico no farmacológico")[:48])
    safe_category = escape(str(category or "Agente tóxico")[:38])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440">
      <defs><linearGradient id="g" x2="1" y2="1">
      <stop stop-color="#17324a"/><stop offset="1" stop-color="#24586b"/></linearGradient></defs>
      <rect width="720" height="440" rx="20" fill="url(#g)"/>
      <circle cx="360" cy="166" r="106" fill="#e8b85c" opacity=".12"/>
      <path d="M360 73 L462 255 L258 255 Z" fill="#ffd67b" stroke="#ffe8b8" stroke-width="9"/>
      <rect x="350" y="130" width="20" height="69" rx="7" fill="#193b50"/>
      <circle cx="360" cy="225" r="12" fill="#193b50"/>
      <text x="360" y="311" font-family="sans-serif" font-size="24" font-weight="bold"
      text-anchor="middle" fill="white">{safe_name}</text>
      <text x="360" y="349" font-family="sans-serif" font-size="18"
      text-anchor="middle" fill="#c4e1ec">{safe_category}</text>
      <text x="360" y="401" font-family="sans-serif" font-size="17"
      text-anchor="middle" fill="#ffd67b">ILUSTRACIÓN ESQUEMÁTICA · NO FOTOGRAFÍA</text>
    </svg>'''.encode("utf-8")


def render_external_toxic_image(row):
    """Se ejecuta solo al abrir una ficha, no al listar las 128 entradas."""
    name = str(row.get("toxico") or "Tóxico")
    category = str(row.get("categoria") or "")
    queries = _queries(row)
    photos = [(term, _commons_photo(term)) for term in queries[:2]]
    usable = [(term, photo) for term, photo in photos if photo]
    st.markdown("#### Imagen identificativa / referencia visual")
    if usable:
        columns = st.columns(len(usable))
        for col, (term, photo) in zip(columns, usable):
            with col:
                st.image(photo["image_url"], caption=f"Referencia: {term}", use_container_width=True)
                label = f"Wikimedia Commons · {photo['license']}"
                st.markdown(f"[{label}]({photo['page_url']})")
                if photo.get("author"):
                    st.caption(f"Autoría: {photo['author'][:100]}")
        if len(usable) < len(queries):
            st.caption("No se encontró una fotografía licenciada y suficientemente identificada de todos los componentes.")
        st.caption(
            "Fotografía de referencia taxonómica. La identificación de una especie "
            "o el diagnóstico de exposición no puede establecerse solo por comparación visual."
        )
    else:
        st.image(_svg_reference(name, category), use_container_width=True)
        st.caption("Ilustración esquemática original; no existe fotografía identificativa verificada para esta ficha.")
    if "TUSI" in _norm(name) or "TUCIBI" in _norm(name):
        st.warning(
            "Tusi o cocaína rosada: mezcla de composición variable. El color o apariencia "
            "no permite identificar sustancias, pureza ni toxicidad. Requiere análisis químico."
        )
