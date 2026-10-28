import json
import re
import time
import urllib.parse
import urllib.request

DOIS = {
    "xiumin2026machine": "10.1038/s41746-026-02565-x",
    "taparugssanagorn2026lightweight": "10.1007/978-3-032-28819-6_15",
    "yong2024deep": "10.1038/s41598-023-49890-9",
    "wang2026deep": "10.2147/rmhp.s629368",
    "hernandezdiaz2026deep": "10.1007/s12065-026-01246-8",
    "swinckels2024deep": "10.2196/48320",
    "lpez2026empirical": "10.48550/arxiv.2603.02719",
    "cenacchi2026beyond": "10.48550/arxiv.2607.18279",
    "strickland2025degrees": "10.1038/s44172-025-00548-6",
    "mehdizavareh2026informative": "10.48550/arxiv.2606.17106",
    "kim2026retamamba": "10.48550/arxiv.2605.16380",
    "shen2026enhancing": "10.48550/arxiv.2607.28035",
    "nie2026clinically": "10.48550/arxiv.2604.20924",
    "fentazi2026training": "10.48550/arxiv.2610.12232",
    "yalavarthi2024grafiti": "10.1609/aaai.v38i15.29560",
    "mun2026made": "10.1093/jamia/ocag124",
    "komolafe2025early": "10.48550/arxiv.2505.02889",
    "yin2024sepsiscalc": "10.1145/3690624.3709402",
    "xu2026bridging": "10.24963/ijcai.2026/733",
    "sun2026review": "10.34133/hds.0456",
    "rong2025deep": "10.1093/jamiaopen/ooaf026",
    "mohammad2024early": "10.48550/arxiv.2402.03486",
    "li2026crosssample": "10.1609/aaai.v40i27.39460",
    "moons2025probastai": "10.1136/bmj-2024-082505",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "refcheck/1.0 (mailto:author@example.org)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def esc(s):
    if s is None:
        return ""
    s = s.replace("\\", "")
    for ch in ["&", "%", "#", "_"]:
        s = s.replace(ch, "\\" + ch)
    return s


def crossref(doi):
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    au = []
    for a in m.get("author") or []:
        g, f = a.get("given", ""), a.get("family", "")
        au.append((g + " " + f).strip() if (g or f) else a.get("name", ""))
    year = (m.get("published") or m.get("issued") or {}).get("date-parts", [[None]])[0][0]
    return {
        "type": m.get("type"),
        "title": (m.get("title") or [""])[0],
        "authors": au,
        "venue": (m.get("container-title") or [""])[0],
        "volume": m.get("volume"),
        "issue": m.get("issue"),
        "page": m.get("page"),
        "year": year,
        "publisher": m.get("publisher"),
        "doi": m.get("DOI", doi),
    }


def datacite(doi):
    m = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi))["data"]["attributes"]
    au = []
    for c in m.get("creators") or []:
        nm = c.get("name") or ((c.get("givenName", "") + " " + c.get("familyName", "")).strip())
        au.append(nm)
    return {
        "type": "preprint",
        "title": (m.get("titles") or [{}])[0].get("title", ""),
        "authors": au,
        "venue": "arXiv",
        "volume": None,
        "issue": None,
        "page": None,
        "year": m.get("publicationYear"),
        "publisher": m.get("publisher"),
        "doi": doi,
    }


# Crossref records for these DOIs are held by DataCite (arXiv DOIs).
ARXIV = {k for k, v in DOIS.items() if v.startswith("10.48550/")}

out = []
for key, doi in DOIS.items():
    rec = None
    try:
        rec = datacite(doi) if key in ARXIV else crossref(doi)
    except Exception as e:
        print("FAIL", key, doi, e)
        continue
    if key in ARXIV:
        out.append(
            "@misc{%s,\n  title = {%s},\n  author = {%s},\n  year = {%s},\n"
            "  howpublished = {arXiv:%s},\n  eprint = {%s},\n  archivePrefix = {arXiv},\n"
            "  doi = {%s},\n  url = {https://doi.org/%s},\n}\n"
            % (key, esc(rec["title"]), " and ".join(rec["authors"]), rec["year"],
               doi.replace("10.48550/arxiv.", ""), doi.replace("10.48550/arxiv.", ""),
               doi, doi)
        )
        continue

    t = rec["type"]
    if t == "proceedings-article":
        head, venue_field = "@inproceedings", "booktitle"
    elif t == "journal-article":
        head, venue_field = "@article", "journal"
    else:
        head, venue_field = "@misc", "howpublished"

    lines = [head + "{" + key + ",", "  title = {%s}," % esc(rec["title"]),
             "  author = {%s}," % " and ".join(rec["authors"]), "  year = {%s}," % rec["year"]]
    if rec["venue"]:
        lines.append("  %s = {%s}," % (venue_field, esc(rec["venue"])))
    for fld, src in (("volume", "volume"), ("number", "issue"), ("pages", "page"),
                     ("publisher", "publisher"), ("doi", "doi")):
        if rec[src]:
            lines.append("  %s = {%s}," % (fld, esc(str(rec[src]))))
    if rec["doi"]:
        lines.append("  url = {https://doi.org/%s}," % rec["doi"])
    lines.append("}\n")
    out.append("\n".join(lines))
    time.sleep(0.2)

with open("refs.bib", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote", len(out), "entries")
