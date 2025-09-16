import regex, json

response_text = """```json
{
 "answer": "En 2022, le nombre d'étudiants inscrits dans la région Vatovavy Fitovinany dans le secteur privé était de 270.",
 "relevant_context": {
  "document": "Pour lann\u00e9e universitaire 2021-2022, 41 des \u00e9tudiants inscrits sont en L1, 23 en L2, 18 en L3, 9 en M1, 7 en M2 et 2 pour les plus de 
6\u00e8me ann\u00e9e. On constate que plus le niveau d\u00e9tude est \u00e9lev\u00e9, le nombre des inscrits devient de plus en plus faible. Tableau 14 : EVOLUTION DES EFFECTIFS DES ETUDIANTS PAR REGION A LENSEIGNEMENT SUPERIEUR DE 2017 A 2022\n\nTableau (format markdown):\n| REGIONS | 2017 |  | 2018 |  | 2019 |  | 2020 |  | 2022 |  |  |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n|  | PUBLIC | PRIVE | PUBLIC | PRIVE | PUBLIC | PRIVE | PUBLIC | PRIVE | PUBLIC | PRIVE |  |\n| Alaotra Mangoro | 389 | 217 | 917 | 315 | 834 | 311 | 992 | 389 | 1344 | 3 29 |  |\n| Amoron'i Mania | 1076 | 353 | 987 | 227 | 638 | 237 | 605 | 211 | 1036 | 2 04 |  |\n| Analanjirofo | 115 | 235 | 119 | 209 | 427 | 317 | 530 | 217 | 704 | 6 7 |  |\n| Analamanga | 48263 | 24705 | 47197 | 26461 | 47 811 | 
29 494 | 46483 | 29256 | 46473 | 3 0 906 |  |\n| Androy | 7 | 0 | 9 | 0 | 393 | 0 | 197 | 0 | 455 | 1 04 |  |\n| Anosy | 113 | 268 | 133 | 258 | 351 | 258 | 423 | 223 | 448 | 1 82 |  |\n| Atsimo Andrefana | 7541 | 748 | 8018 | 494 | 8 250 | 589 | 8777 | 580 | 11177 | 5 69 |  |\n| Atsimo Atsinanana | 78 | 272 | 85 | 158 | 82 | 158 | 72 | 158 | 211 | 1 58 |  |\n| Atsinanana | 9089 | 915 | 8137 | 946 | 8 381 | 966 | 10061 | 997 | 12309 | 7 22 |  |\n| Betsiboka | 36 | 0 | 47 | 0 | 32 | 0 | 80 | 0 | 118 | 0 |  |\n| Boeny | 7676 | 1656 | 8011 | 1888 | 8 572 | 1 852 | 10673 | 2263 | 12548 | 2 361 |  |\n| Bongolova | 168 | 260 | 147 | 281 | 146 | 294 | 128 | 123 | 136 | 1 96 |  |\n| DIANA | 6765 | 557 | 7325 | 878 | 8 499 | 795 | 9421 | 848 | 10712 | 6 44 |  |\n| Haute Matsiatra | 18036 | 1574 | 20445 | 1453 | 22 111 | 1 535 | 23059 | 1241 | 29184 | 2 330 |  |\n| Ihorombe | 60 | 77 | 68 | 91 | 60 | 99 | 53 | 98 | 84 | 1 18 |  |\n| Itasy | 315 | 196 | 819 | 205 | 1 009 | 185 | 1204 | 190 | 1657 | 2 89 |  |\n| Melaky | 45 | 0 | 36 | 0 | 43 | 0 | 38 | 0 | 63 | - |  |\n| Menabe | 81 | 320 | 88 | 297 | 71 | 293 | 186 | 224 | 302 | 8 0 |  |\n| SAVA | 140 | 626 | 152 | 853 | 123 | 361 | 570 | 177 | 988 | 1 64 |  |\n| Sofia | 186 | 506 | 264 | 534 | 160 | 542 | 1088 | 641 | 1562 | 4 34 |  |\n| Vakinankaratra | 912 | 3177 | 1169 | 3602 | 2 363 | 3 761 | 2633 | 3668 | 2730 | 4 064 |  |\n| Vatovavy Fitovinany | 141 | 0 | 155 | 281 | 133 | 307 | 212 | 270 | 233 | 270 |  |\n| TOTAL | 101 232 | 36 662 | 104 328 | 39 431 | 110 489 | 42 354 | 117 485 | 41 774 | 134 474 | 44 191 |  |\n\n\nSource : Annuaire Statistique SSP DSSIP MESUPRES 19",
  "metadata": {
   "logical_page": 24,
   "filename": "MESUPRES_en_chiffres_MAJ.pdf",
   "half": "right",
   "physical_page": 13
  }
 }
}
```"""
json_match = regex.search(r'\{(?:[^{}]|(?R))*\}', response_text, regex.DOTALL)
if json_match:
    json_str = json_match.group(0)

    json_str = json_str.replace(",}", "}").replace(",]", "]")
    json_str = json_str.replace("\n", "  ")

    print(json_str)  # debug

    data = json.loads(json_str)
    print(data["answer"])