# Detekcja anomalii i ataków w żądaniach HTTP — CSIC 2010

Projekt dotyczy oceny skuteczności modeli uczenia maszynowego w detekcji anomalii i potencjalnych ataków w żądaniach HTTP.

## Dataset

W projekcie wykorzystano dataset **CSIC 2010 Web Application Attacks**.

Plik z danymi znajduje się w katalogu:

```text
data/csic_database.csv

Dataset zawiera żądania HTTP opisane m.in. przez metodę HTTP, URL, treść żądania oraz etykietę klasyfikacyjną.

Problem został sprowadzony do klasyfikacji binarnej:

0 — ruch normalny
1 — anomalia / potencjalny atak
Modele

W projekcie wykorzystano trzy modele:

Random Forest — model nadzorowany, trenowany na próbkach normalnych i anomalnych.
Isolation Forest — model detekcji anomalii, trenowany wyłącznie na próbkach normalnych.
One-Class SVM — model one-class, trenowany wyłącznie na próbkach normalnych, z użyciem StandardScaler.
Zestawy cech

W projekcie przygotowano dwa zestawy cech.

BASIC
url_len
content_len
url_special_chars
content_special_chars
ADVANCED
url_len
content_len
url_special_chars
content_special_chars
sqli_flag
xss_flag
traversal_flag

Finalne wyniki przyjęto dla wariantu BASIC, ponieważ nie wykorzystuje on ręcznie zdefiniowanych flag ataków.

Struktura projektu
project/
├── data/
│   └── csic_database.csv
├── models/
├── reports/
├── wyniki/
├── config.py
├── data_processor.py
├── model_rand_forest.py
├── model_iforest.py
├── model_ocsvm.py
├── evaluator.py
├── requirements.txt
└── README.md
Pliki wynikowe

Najważniejsze gotowe wyniki znajdują się w katalogu:

wyniki/

Po uruchomieniu ewaluacji dodatkowe wyniki generowane są w katalogu:

reports/

W katalogu reports/ zapisywane są:

metrics_summary.csv
confusion_matrices.png
roc_curves.png
Instalacja zależności

Zalecane jest utworzenie środowiska wirtualnego:

python -m venv .venv

Aktywacja środowiska w Windows:

.venv\Scripts\activate

Instalacja bibliotek:

pip install -r requirements.txt

Przykładowa zawartość requirements.txt:

pandas
scikit-learn
matplotlib
seaborn
joblib
Uruchomienie projektu

Projekt należy uruchamiać z głównego katalogu projektu.

Najpierw należy wytrenować modele:

python model_rand_forest.py
python model_iforest.py
python model_ocsvm.py

Następnie należy uruchomić ewaluację:

python evaluator.py

Pełna kolejność uruchamiania:

python model_rand_forest.py
python model_iforest.py
python model_ocsvm.py
python evaluator.py