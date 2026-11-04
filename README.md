# SnowHybrid + ECMWF — twee modeltabs op GitHub Pages

Website: https://landervt.github.io/weather-maps-sh/

Met deze versie draait de bestaande `SnowHybrid.py` voor 48 uur **EN** `SnowDays_ECMWF.py` afzonderlijk voor 7 dagen. De website krijgt twee tabbladen: **Hybrid Model** en **ECMWF Model**.

## Drie wijzigingen in je bestaande repository

De originele zeven wetenschappelijke scripts blijven ongemoeid. Je hoeft **slechts drie bestanden** toe te voegen/vervangen:

1. **Vervang** `run_forecast.py` in de hoofdmap met de versie uit dit pakket. Deze bouwt de website met twee tabs en verwijdert oudere kaarten.
2. **Voeg toe** `run_ecmwf.py` in de hoofdmap. Dit is de aparte runner voor de originele `SnowDays_ECMWF.py`.
3. **Vervang** `.github/workflows/update.yml` door de nieuwe YAML uit dit pakket. Let op: de `.github`-map is op Mac verborgen (Cmd + Shift + punt).

Via GitHub.com: ga naar **Code**, open `run_forecast.py`, klik op het potloodje en vervang de volledige inhoud. Voor `run_ecmwf.py` kies **Add file > Upload files**. Open daarna via **Code > .github > workflows > update.yml** en vervang de inhoud met het potloodje. Sla wijzigingen telkens op met **Commit changes**.

Je kunt ook GitHub Desktop gebruiken: clone de repository, kopieer alle bestanden uit het ZIP-pakket (inclusief verborgen `.github`) over de bestaande bestanden, klik **Commit to main** en daarna **Push origin**.

**Let op:** de nieuwe workflow heeft bewust geen `push`-trigger meer. Elke kleine codewijziging hoeft geen weermodeldownload te starten. De website blijft wel ieder uur ingepland en je kunt altijd handmatig starten.

## De eerste echte test

1. Open **Actions > Update SnowHybrid + ECMWF maps (hourly)**.
2. Klik op **Run workflow** en vink eerst `check_only` aan. Een groene check betekent dat alle scripts en imports beschikbaar zijn; er worden geen voorspellingen gemaakt.
3. Start nog een run met `check_only` **uitgevinkt**.
4. Onder **Generate Hybrid + standalone ECMWF maps and website** zie je eerst Hybrid, daarna ECMWF. Bij succes worden beide sets gepubliceerd.
5. Bekijk https://landervt.github.io/weather-maps-sh/ en klik bovenaan **Hybrid Model** of **ECMWF Model**. De tabkeuze kan ook via URL-fragment `#ecmwf` of `#hybrid`.

## Kaarten en schijfruimte

- **Hybrid Model**: originele 6-models berekening tot +48 uur; dagkaarten, cumulatieve sneeuw, sneeuw op de grond, sneeuwkans en modelonenigheid.
- **ECMWF Model**: de **afzonderlijke** 7-daagse berekening via `SnowDays_ECMWF.py`, niet het ECMWF-aandeel in Hybrid. Voor de Alpen maakt het originele script 7 dagkaarten, 6 cumulatieve kaarten en 7 kaarten met nieuwe sneeuw op de grond: **20 PNG's**.
- Modeldata, DEM, kaarttegels en tijdelijke arrays worden tijdens een run opgeslagen onder `_run/` op de GitHub-server. Na afloop verwijdert `run_forecast.py` alle tijdelijke bestanden.
- De website bevat enkel nieuwe PNG's onder `site/maps/hybrid/` en `site/maps/ecmwf/`. De site wordt iedere publicatie volledig opnieuw opgebouwd. Oude kaarten blijven dus niet staan.
- Als een model faalt, toont dat tabblad een foutmelding en geen oude kaarten. Een geslaagde andere modelrun blijft wel zichtbaar. De workflow wordt rood om het probleem kenbaar te maken als een van de twee modellen mislukt.
- Zonder nieuwe publicatie worden kaarten na zes uur op de site automatisch verborgen als verouderd.

## Instellingen in `.github/workflows/update.yml`

| Variabele | In deze versie | Betekenis |
| --- | --- | --- |
| `HYBRID_GRID_STEP` | `0.008` | Hybrid, lagere resolutie voor een stabiele cloud-run; originele waarde `0.004` |
| `HYBRID_DPI` | `200` | Hybrid-figuren; origineel `400` |
| `ECMWF_AREAS` | `alps` | Veilige eerste test; `alps,full` maakt ook West-Europakaarten (aanzienlijk zwaarder) |
| `ECMWF_N_DAYS` | `7` | Originele ECMWF-voorspelling van 7 dagen |
| `ECMWF_GRID_STEP_ALPS` | `0.008` | ECMWF-doelrooster Alpen |
| `ECMWF_GRID_STEP_FULL` | `0.016` | Grover rooster voor eventueel West-Europa |
| `ECMWF_DPI` | `200` | ECMWF-kaarten |
| `ECMWF_ESRI_RELIEF` | `1` | Reliëf aan; bij tegelproblemen tijdelijk `0` |

**West-Europa ook publiceren?** Wijzig na een geslaagde test `ECMWF_AREAS: 'alps'` naar `ECMWF_AREAS: 'alps,full'`. Dat verdubbelt het aantal ECMWF-kaarten (20 + 20 = 40) en verhoogt downloads, geheugen en rekentijd. Een volle 400m-`full`-kaart is voor GitHub Actions mogelijk te zwaar, daarom heeft `ECMWF_GRID_STEP_FULL` een eigen grovere instelling.

## Automatische planning

De workflow gebruikt `cron: '07 * * * *'`: GitHub probeert elk uur rond minuut 07 UTC een run te starten. Er is geen garantie dat die precies op tijd begint, en geplande runs kunnen vertragen of worden overgeslagen. Je kunt een handmatige run starten via **Run workflow**. De concurrency-groep heeft een nieuwe naam (`snowhybrid-ecmwf-pages-v3`) zodat eerder vastgelopen runs uit oudere workflowversies geen blokkade vormen.

De workflow heeft een limiet van 55 minuten per uitvoering. Indien ECMWF met 7 dagen te zwaar blijkt, probeer dan `ECMWF_N_DAYS: '3'`, verlaag de DPI, zet reliëf uit of gebruik een krachtigere runner.

## Oorspronkelijke modelcode

Dit pakket bevat de zeven originele modelbestanden, identiek aan de eerder werkende GitHub-versie:

- `SnowHybrid.py`
- `SnowDays_AROME.py`
- `SnowDays_AROME-AT.py`
- `SnowDays_ICON-CH1.py`
- `SnowDays_ICON-CH2.py`
- `SnowDays_ICON-D2.py`
- `SnowDays_ECMWF.py`

Zij worden niet aangepast: de aparte ECMWF-runner importeert het oorspronkelijke script, configureert de uitvoermap en cloudresolutie, en roept `main()` aan. Een bestaand eigen `logo-sneeuwhoogte.png` in de repository blijft worden gebruikt; anders verschijnt tijdelijk een neutraal logo op beide kaarttypes.

**Teststatus:** Python-syntax, websitevervanging, tabs, ontbrekende-modelafhandeling en de ECMWF-runner met een gesimuleerde modelbron zijn lokaal getest. Een echte downloadsessie voor ECMWF en de GitHub-runnerbelasting kan pas via de eerste echte Actions-run worden gecontroleerd.
