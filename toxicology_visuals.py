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
# Fotografía con atribución/licencia comprobada en la página de Commons.
PINNED_COMMONS = {
    "Loxosceles laeta": {
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Loxosceles.laeta.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Loxosceles.laeta.jpg",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "author": "Mampato",
        "scientific": "Loxosceles laeta",
    },
}

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
    if query in PINNED_COMMONS:
        return PINNED_COMMONS[query]
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
    """Ilustración original; nunca pretende identificar especie por morfología."""
    safe_name = escape(str(name or "Tóxico no farmacológico")[:48])
    safe_category = escape(str(category or "Agente tóxico")[:38])
    label = _norm(name)
    cat = _norm(category)

    if "TUSI" in label or "TUCIBI" in label or "COCAINA ROSADA" in label:
        motif = """
        <ellipse cx="360" cy="250" rx="150" ry="33" fill="#122c43" opacity=".45"/>
        <path d="M210 239 Q255 202 294 166 Q320 153 348 178 Q378 132 404 168
          Q451 175 511 240 Q393 281 210 239 Z" fill="#e76aaf" stroke="#f0a6ce" stroke-width="4"/>
        <g fill="#ffd3e7">
          <circle cx="299" cy="204" r="5"/><circle cx="339" cy="184" r="4"/>
          <circle cx="386" cy="207" r="7"/><circle cx="428" cy="230" r="4"/>
          <circle cx="358" cy="233" r="5"/><circle cx="273" cy="232" r="4"/>
        </g>"""
        sub = "REPRESENTACIÓN DE POLVO ROSA · COMPOSICIÓN DESCONOCIDA"
    elif cat in ("PLANTAS",):
        motif = """
        <path d="M350 281 Q343 180 360 95" fill="none" stroke="#b6d989" stroke-width="13" stroke-linecap="round"/>
        <path d="M350 209 Q280 124 241 153 Q261 226 350 209 M352 158 Q438 81 480 122
          Q449 194 352 158" fill="#7ecb91" stroke="#c7f3bd" stroke-width="3"/>
        <circle cx="318" cy="241" r="18" fill="#f3e1c5"/><circle cx="365" cy="245" r="19" fill="#f3e1c5"/>
        <circle cx="391" cy="213" r="15" fill="#f3e1c5"/>"""
        sub = "ILUSTRACIÓN BOTÁNICA ESQUEMÁTICA · NO IDENTIFICACIÓN"
    elif cat in ("ARACNIDOS", "ESCORPIONES", "ARTROPODOS", "ORUGAS"):
        motif = """
        <ellipse cx="360" cy="192" rx="40" ry="32" fill="#bb995f"/>
        <ellipse cx="358" cy="247" rx="50" ry="58" fill="#b4864a"/>
        <g stroke="#dcb67c" stroke-width="10" stroke-linecap="round" fill="none">
        <path d="M322 185 L258 147 L211 113 M322 201 L258 187 L194 192
        M326 224 L263 257 L215 300 M332 239 L300 302 L276 344
        M398 185 L464 147 L510 113 M398 201 L459 187 L526 192
        M398 224 L463 257 L511 300 M387 239 L419 302 L444 344"/></g>"""
        sub = "ILUSTRACIÓN ARTRÓPODO · NO IDENTIFICACIÓN DE ESPECIE"
    elif cat in ("SERPIENTES", "REPTILES"):
        motif = """
        <path d="M240 207 Q302 123 366 209 T473 222 Q534 199 473 152" fill="none"
          stroke="#8cc6a1" stroke-width="31" stroke-linecap="round"/>
        <circle cx="469" cy="151" r="8" fill="#17324a"/>
        <path d="M483 161 L503 174" stroke="#ef8b8b" stroke-width="5"/>"""
        sub = "ILUSTRACIÓN REPTIL · NO IDENTIFICACIÓN DE ESPECIE"
    elif cat == "HONGOS":
        motif = """
        <path d="M340 209 L318 296 Q355 316 391 296 L375 209 Z" fill="#f4e2be"/>
        <path d="M242 218 Q258 107 359 107 Q463 108 478 218 Z" fill="#d87d5e"/>
        <circle cx="326" cy="159" r="14" fill="#ffe0b5"/><circle cx="391" cy="184" r="11" fill="#ffe0b5"/>
        <path d="M241 218 Q359 242 478 218" stroke="#e8ae8a" stroke-width="5" fill="none"/>"""
        sub = "ILUSTRACIÓN HONGO · NO IDENTIFICACIÓN DE ESPECIE"
    elif cat in ("MARINOS", "TOXINAS MARINAS"):
        motif = """
        <path d="M247 215 Q334 112 464 216 Q334 320 247 215
          L186 160 L186 271 Z" fill="#8bd1d9" stroke="#c6f2f2" stroke-width="6"/>
        <circle cx="418" cy="203" r="12" fill="#163a4a"/>
        <path d="M339 195 L339 238" stroke="#347e89" stroke-width="5"/>"""
        sub = "ILUSTRACIÓN MARINA · NO IDENTIFICACIÓN DE ESPECIE"
    else:
        motif = """
        <path d="M360 78 L474 262 L246 262 Z" fill="#ffd67b" stroke="#ffe9ba" stroke-width="8"/>
        <rect x="351" y="142" width="18" height="67" rx="7" fill="#183b50"/>
        <circle cx="360" cy="231" r="12" fill="#183b50"/>"""
        sub = "ESQUEMA DEL AGENTE · NO FOTOGRAFÍA"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440">
      <defs><linearGradient id="g" x2="1" y2="1">
      <stop stop-color="#17324a"/><stop offset="1" stop-color="#24586b"/></linearGradient></defs>
      <rect width="720" height="440" rx="20" fill="url(#g)"/>
      {motif}
      <text x="360" y="330" font-family="sans-serif" font-size="22" font-weight="bold"
      text-anchor="middle" fill="white">{safe_name}</text>
      <text x="360" y="360" font-family="sans-serif" font-size="17"
      text-anchor="middle" fill="#c4e1ec">{safe_category}</text>
      <text x="360" y="406" font-family="sans-serif" font-size="16"
      text-anchor="middle" fill="#ffd67b">{escape(sub)}</text>
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
