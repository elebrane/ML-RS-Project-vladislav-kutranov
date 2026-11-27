"""Alias table for the candidate stocks: GDELT query phrases (English, searched in GDELT's translation)
and title patterns (Greek or Latin, matched on the original-language headline; '*' = word prefix).

Manual entries were written from public company names (2021 names and later renames); the rest are
generated from assetName. Precision is checked by hand on a sample (news/linking_check_sample.csv).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = {
 'GRS145003000': (['GEK Terna'], ['γεκ τερνα', 'gek terna', 'γεκτερνα']),
 'GRS496003005': (['Terna Energy'], ['τερνα ενεργειακ*', 'terna energy', 'τερνα energy']),
 'BE0974303357': (['Cenergy'], ['cenergy']),
 'GRS087003000': (['Intracom Holdings', 'Intracom'], ['intracom', 'ιντρακομ']),
 'GRS359353000': (['EYDAP', 'Athens Water Supply'], ['ευδαπ', 'eydap']),
 'GRS314003005': (['Marfin Investment Group'], ['marfin', 'μιγ']),
 'GRS191213008': (['Ellaktor'], ['ελλακτωρ*', 'ellaktor']),
 'GRS260333000': (['Hellenic Telecommunications', 'Cosmote', 'OTE Group'], ['οτε', 'ote', 'cosmote']),
 'GRS419003009': (['OPAP'], ['οπαπ', 'opap']),
 'GRS426003000': (['Motor Oil Hellas', 'Motor Oil'], ['motor oil', 'μοτορ οιλ']),
 'GRS518003009': (['ADMIE', 'Independent Power Transmission Operator'], ['αδμηε', 'admie', 'ipto']),
 'GRS393503008': (['Mytilineos'], ['μυτιληναι*', 'mytilineos']),
 'GRS281003004': (['ElvalHalcor', 'Elval Halcor'], ['ελβαλχαλκορ', 'elvalhalcor', 'elval']),
 'GRS434003000': (['Public Power Corporation'], ['δεη', 'ppc']),
 'GRS298343005': (['Hellenic Petroleum', 'Helleniq Energy'], ['ελπε', 'ελληνικα πετρελαια', 'hellenic petroleum', 'helleniq']),
 'GRS323003012': (['Eurobank'], ['eurobank']),
 'GRS239003007': (['Thrace Plastics'], ['πλαστικα θρακης', 'thrace plastics']),
 'GRS395363005': (['Hellenic Exchanges', 'ATHEX Group'], ['ελληνικα χρηματιστηρια', 'εχαε', 'athex group']),
 'GRS213213002': (['Avax'], ['αβαξ', 'avax']),
 'GRS014003032': (['Piraeus Bank', 'Piraeus Financial Holdings'], ['τραπεζα πειραιως', 'πειραιως financial', 'πειραιως holdings', 'piraeus bank', 'piraeus financial']),
 'GRS495003006': (['Aegean Airlines'], ['aegean airlines', 'αεροπορια αιγαιου', 'aegean']),
 'GRS346003007': (['Frigoglass'], ['frigoglass']),
 'GRS372003004': (['Iktinos Hellas', 'Iktinos'], ['ικτινος', 'iktinos']),
 'GRS015003007': (['Alpha Bank', 'Alpha Services and Holdings'], ['alpha bank', 'alpha services', 'alpha τραπεζα']),
 'GRS245213004': (['Lamda Development'], ['lamda']),
 'GRS003003035': (['National Bank of Greece'], ['εθνικη τραπεζα', 'εθνικης τραπεζας', 'national bank of greece', 'nbg']),
 'GRS004013009': (['Bank of Greece'], ['τραπεζα της ελλαδος', 'τραπεζας της ελλαδος', 'bank of greece']),
 'GRS001003037': (['Attica Bank'], ['attica bank', 'τραπεζα αττικης', 'τραπεζας αττικης']),
 'GRS096003009': (['Fourlis'], ['φουρλη*', 'fourlis']),
 'GRS282183003': (['Jumbo'], ['jumbo']),
 'CY0109232112': (['Cairo Mezz'], ['cairo mezz']),
 'GRS368313003': (['Byte Computer'], ['byte computer']),
 'GRS472003011': (['Profile Systems', 'Profile Software'], ['profile systems', 'profile software']),
 'GRS505003004': (['Performance Technologies'], ['performance technologies']),
 'GRS289103004': (['Alumil'], ['alumil']),
 'GRS475003018': (['Ilyda'], ['ilyda', 'ιλυδα']),
 'GRS461003006': (['Logismos'], ['logismos']),
 'GRS144003001': (['Attica Group', 'Attica Holdings'], ['attica group', 'attica holdings', 'attica συμμετοχων']),
 'GRS204003008': (['Sarantis'], ['σαραντη*', 'sarantis']),
 'GRS477003008': (['Elinoil'], ['elinoil', 'ελινοιλ']),
 'GRS106003007': (['Reds Real Estate'], ['reds']),
 'GRS439003005': (['Euroconsultants'], ['euroconsultants']),
 'GRS498003003': (['Epsilon Net'], ['epsilon net']),
 'GRS510003015': (['Foodlink'], ['foodlink']),
 'GRS147233001': (['Athens Medical Center', 'Iatriko Athinon'], ['ιατρικο αθηνων', 'athens medical']),
 'GRS295003008': (['Kiriacoulis'], ['kiriacoulis', 'κυριακουλη*']),
 'GRS332073006': (['Fieratex'], ['fieratex']),
 'GRS384003000': (['Kordellos'], ['kordellos', 'κορδελλο*']),
 'GRS345503007': (['Petros Petropoulos'], ['petropoulos', 'πετροπουλο*']),
 'GRS277023008': (['European Reliance'], ['european reliance', 'ευρωπαικη πιστη']),
 'GRS517003000': (['BriQ Properties'], ['briq']),
 'GRS427003009': (['Thessaloniki Port Authority', 'ThPA'], ['ολθ', 'thpa', 'λιμενος θεσσαλονικης']),
 'GRS473003002': (['Revoil'], ['revoil']),
 '100974271034': (['Viohalco'], ['viohalco', 'βιοχαλκο']),
 'GRS070003009': (['Kekrops'], ['kekrops', 'κεκροψ']),
}


def auto(name):
    words = [w for w in re.split(r'[^A-Za-z]+', name or '') if w and w.upper() not in
             {'SA', 'S', 'A', 'CR', 'SR', 'HOLDINGS', 'HOLDING', 'COMPANY', 'GROUP', 'PLC', 'THE', 'AND', 'OF'}]
    phrase = ' '.join(words[:2]).title()
    return ([phrase], [phrase.lower()]) if phrase else ([], [])


def main():
    stocks = json.loads((HERE / 'news' / 'candidate_stocks.json').read_text())
    rows = []
    for s in stocks:
        q, t = M.get(s['isin']) or auto(s['name'])
        rows.append({'isin': s['isin'], 'name': s['name'], 'ticker_gr': s['short'], 'query_phrases': q,
                     'title_patterns': t, 'source': 'manual' if s['isin'] in M else 'auto',
                     'weight': sum(s['clients'].values())})
    (HERE / 'news' / 'aliases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
    print(len(rows), 'assets;', sum(r['source'] == 'manual' for r in rows), 'manual')


if __name__ == '__main__':
    main()
