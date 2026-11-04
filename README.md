# SnowHybrid: automatisch elk uur sneeuwkaarten via GitHub Actions

Website: https://landervt.github.io/weather-maps-sh/
Repository: https://github.com/LanderVT/weather-maps-sh

## Inhoud

- `SnowHybrid.py`: de hybride sneeuwberekening (zes modellen, tot +48 uur)
- `SnowDays_AROME.py`, `SnowDays_AROME-AT.py`, `SnowDays_ICON-CH1.py`, `SnowDays_ICON-CH2.py`, `SnowDays_ICON-D2.py`, `SnowDays_ECMWF.py`: de originele zes model-downloaders, inhoudelijk ongewijzigd
- `run_forecast.py`: tijdelijke werkmap, logofallback, nieuwe kaarten, website en opruimen
- `requirements.txt`: de benodigde Python-pakketten
- `.github/workflows/update.yml`: workflow voor de automatische uitvoering
- `.gitignore`: voorkomt dat data of kaarten ooit onbedoeld gecommit worden

## GitHub-installatie (via de website)

1. Download deze ZIP en pak ze op je Mac uit. **Upload de ZIP niet letterlijk naar GitHub**: GitHub voert de bestanden in de ZIP niet automatisch uit.
2. Open de repository `LanderVT/weather-maps-sh` en ga naar **Code > Add file > Upload files**.
3. Upload de **acht Python-bestanden** (`SnowHybrid.py`, de zes `SnowDays_*.py` en `run_forecast.py`) plus `requirements.txt` en eventueel `README.md` naar de **hoofdfolder** (`main`). Gebruik de exacte originele namen, zonder `(2)`.
4. Open in GitHub het bestaande `.github/workflows/update.yml` en klik op het potloodje. Vervang de hele inhoud door de inhoud van `.github/workflows/update.yml` uit deze ZIP. Commit naar `main`.
5. Ga naar **Settings > Pages** en controleer dat **Source = GitHub Actions** staat (bij jou was dit al ingesteld).
6. Ga naar **Actions > Update SnowHybrid maps (hourly) > Run workflow**. Vink eerst **check_only** aan voor een snelle test van imports en libraries. Deze test downloadt niets en verandert de website niet.
7. Start vervolgens **Run workflow** opnieuw zonder `check_only`: dan worden de echte weerdata opgehaald en kaarten gemaakt en gepubliceerd.
8. Kijk naar https://landervt.github.io/weather-maps-sh/ en bij problemen naar de rode stap onder **Actions > forecast**.

Een ander eenvoudig alternatief is GitHub Desktop: clone de repository, kopieer de **inhoud** van de uitgepakte ZIP naar de lokale repositorymap (ook `.github`, op Mac zichtbaar met `Cmd + Shift + .`), en kies **Commit to main** en **Push origin**. Daarna dezelfde Actions-tests.

**Let op:** upload het optionele eigen `logo-sneeuwhoogte.png` naar de repository-root als je het originele logo op elke kaart wilt. Zonder dit bestand wordt een duidelijk neutraal `HYBRID`-label gegenereerd.

De oude testbestanden `generate.py` en de oude README mogen blijven, maar `generate.py` wordt door deze nieuwe workflow niet meer gebruikt. Je kunt het later verwijderen.

## Kaarten en modeldata automatisch verversen

- Elke forecast start in een lege `_run/`-werkmap. Data worden opnieuw bij de officiële modelbronnen opgehaald.
- De modellen houden downloads, DEM-tegels en Esri-relieftegels uitsluitend in een **tijdelijke** werkmap onder `_run/`.
- `SnowHybrid.py` maakt de kaarten in `_run/maps/`, en slaat de modelrun-tijden op in `forecast_metadata.json`.
- `run_forecast.py` publiceert **uitsluitend de nieuwe PNG's** en de huidige informatie in `site/`.
- Na de berekening wordt de tijdelijke werkmap met data en oorspronkelijke kaartbestanden verwijderd; alleen de actuele website-artefacten bestaan tijdens de publicatie. Een nieuwe publicatie vervangt de gehele vorige website, inclusief oude PNG's.
- Wanneer de berekening mislukt, wordt een foutpagina **zonder oude kaarten** gepubliceerd en krijgt de workflow uiteindelijk een rood kruisje. Als GitHub Pages zelf of een eerdere installatiestap niet werkt, kan de oude website nog tijdelijk online blijven; de ingebouwde actualiteitscontrole verbergt kaarten zodra ze meer dan zes uur oud zijn.
- De runner is na elke job weg; datasets of kaarten worden dus niet bewaard in de repository.

## Instellingen

Open `.github/workflows/update.yml` en verander indien nodig:

| Variabele | Eerste test | Origineel | Opmerking |
| --- | --- | --- | --- |
| `HYBRID_GRID_STEP` | `0.008` | `0.004` | 0.008 = grovere doelgrid om GitHub niet te overbelasten; 0.004 = originele ~300–450 m grid |
| `HYBRID_DPI` | `200` | `400` | Lager = kleinere PNG's en minder rekentijd |
| `HYBRID_MIN_MODELS` | `2` | `2` | Minimaal twee beschikbare modellen per 6-uursblok |
| `HYBRID_ESRI_RELIEF` | `1` | `1` | Zet op `0` als Esri-tegels niet downloaden; de kaarten krijgen dan een vlakke ondergrond |

**Belangrijk:** de eerste test behoudt alle zes modellen, alle rekenstappen en alle soorten kaarten, maar gebruikt een iets grovere outputgrid en lagere beeldresolutie. Zodra de eerste echte uitvoering geslaagd is, kun je beide op de oorspronkelijke waarden zetten.

## Rekentijd en beperkingen

De oorspronkelijke scripts gebruiken veel geheugen, GRIB-data en Cartopy-achtergrondbeelden. Een publieke repository gebruikt de standaard GitHub Actions Ubuntu-runner met doorgaans 4 cores, 16 GB RAM en 14 GB opslag. De workflow stopt na 55 minuten als hij te lang duurt. Bij de eerste echte uitvoering kunnen API-wijzigingen, downloadlimieten of ontbrekende externe kaarttegels zichtbaar worden; dat kan alleen met een live GitHub-run worden getest.

De planning `17 * * * *` vraagt een update ieder uur rond minuut 17 UTC. GitHub garandeert de exacte start niet. Als de volledige run te zwaar is, probeer een grovere grid, minder DPI, minder runs per dag, of een cloudserver.

Publiceer geen wachtwoorden, API-sleutels of andere privégegevens in deze publieke repository. Voor eventuele API-credentials gebruikt GitHub **Settings > Secrets and variables > Actions**.
