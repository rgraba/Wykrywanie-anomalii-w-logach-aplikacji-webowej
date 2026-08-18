# Wykrywanie anomalii w logach aplikacji webowej

Projekt stanowi część praktyczną pracy magisterskiej poświęconej wykrywaniu anomalii i potencjalnych ataków na podstawie analizy żądań HTTP. Celem projektu jest porównanie klasycznych metod uczenia maszynowego oraz metod jednoklasowych na zbiorze CSIC 2010, a także zbadanie wpływu reprezentacji danych, doboru cech, selekcji cech i strojenia hiperparametrów na skuteczność detekcji.

W aktualnym zakresie badawczym wykorzystywane są trzy modele:

* Random Forest — model nadzorowany uczony na próbkach normalnych i anomalnych;
* Isolation Forest — model detekcji anomalii uczony wyłącznie na próbkach normalnych;
* One-Class SVM — model jednoklasowy uczony wyłącznie na próbkach normalnych.

Projekt obejmuje eksperymenty oparte zarówno na ręcznie skonstruowanych cechach liczbowych, jak i na tekstowej reprezentacji żądań HTTP z wykorzystaniem TF-IDF, znakowych n-gramów oraz redukcji wymiarowości Truncated SVD.

## Zbiór danych

W projekcie wykorzystano zbiór **CSIC 2010 Web Application Attacks**. Plik wejściowy znajduje się w lokalizacji:

```text
data/csic\\\\\\\_database.csv
```

Wymagane kolumny:

* `Method` — metoda HTTP;
* `URL` — adres lub ścieżka żądania;
* `content` — treść żądania;
* `classification` — etykieta klasy.

Problem został sprowadzony do klasyfikacji binarnej:

* `0` — ruch normalny;
* `1` — anomalia lub potencjalny atak.

Aktualny zbiór zawiera 61 065 rekordów:

* 36 000 próbek normalnych;
* 25 065 próbek anomalnych.

## Metodologia podziału danych

W projekcie występują dwa rodzaje podziału danych.

### Podział losowy

Podział losowy jest zachowany w eksperymentach bazowych i eksploracyjnych. Pozwala odtworzyć wcześniejsze wyniki oraz pokazać wpływ duplikatów żądań na ocenę modeli. Wyniki takich eksperymentów są oznaczone wartością:

```text
split = random
```

### Podział grupowy

Podział grupowy jest podstawowym protokołem końcowej oceny modeli. Każde żądanie otrzymuje identyfikator grupy będący skrótem SHA-256 kolumny `request\\\\\\\_text`, zbudowanej z metody HTTP, adresu URL i treści żądania. Identyczne żądania trafiają dzięki temu wyłącznie do jednej części danych.

Podział grupowy ogranicza zawyżanie wyników spowodowane występowaniem identycznych rekordów jednocześnie w treningu i teście. Kod kontroluje liczbę wspólnych grup i dla poprawnego podziału wymaga:

```text
group\\\\\\\_overlap = 0
```

Zapisany, odtwarzalny podział znajduje się w pliku:

```text
splits/group\\\\\\\_split\\\\\\\_seed\\\\\\\_42.csv
```

Grupowanie eliminuje nakładanie się dokładnie identycznych żądań. Nie jest ono równoznaczne z grupowaniem semantycznie podobnych, ale nieidentycznych żądań.

## Reprezentacje danych i zestawy cech

### BASIC

Podstawowy zestaw czterech cech liczbowych:

* `url\\\\\\\_len`;
* `content\\\\\\\_len`;
* `url\\\\\\\_special\\\\\\\_chars`;
* `content\\\\\\\_special\\\\\\\_chars`.

### ALTHUBITI\_9

Zestaw dziewięciu cech odtworzonych na podstawie rozwiązania opisanego w literaturze:

* `request\\\\\\\_length`;
* `arguments\\\\\\\_length`;
* `arguments\\\\\\\_count`;
* `arguments\\\\\\\_digit\\\\\\\_count`;
* `path\\\\\\\_length`;
* `arguments\\\\\\\_letter\\\\\\\_count`;
* `path\\\\\\\_letter\\\\\\\_count`;
* `path\\\\\\\_special\\\\\\\_char\\\\\\\_count`;
* `max\\\\\\\_request\\\\\\\_byte`.

### ALTHUBITI\_5

Pięcioelementowy podzbiór cech literaturowych:

* `request\\\\\\\_length`;
* `arguments\\\\\\\_length`;
* `arguments\\\\\\\_count`;
* `path\\\\\\\_length`;
* `path\\\\\\\_special\\\\\\\_char\\\\\\\_count`.

### ADVANCED

Opcjonalny zestaw rozszerzony zawiera cechy BASIC oraz flagi wzorców SQL Injection, XSS i Path Traversal. Nie jest on domyślnym zestawem cech modeli bazowych i nie stanowi podstawy końcowego porównania.

### Reprezentacja tekstowa

Kolumna `request\\\\\\\_text` powstaje przez połączenie:

```text
Method + URL + content
```

Jest ona wykorzystywana przez modele TF-IDF. Random Forest pracuje bezpośrednio na znakowych n-gramach TF-IDF, natomiast Isolation Forest i One-Class SVM wykorzystują dodatkowo Truncated SVD i standaryzację.

## Selekcja cech

Projekt implementuje trzy metody selekcji cech:

* **Information Gain** — wykorzystuje informację wzajemną; wybierane są cechy z wynikiem większym od średniej;
* **L1** — wykorzystuje pipeline `StandardScaler` + `LogisticRegression` (`l1\_ratio=1.0) oraz `GridSearchCV` z 5-krotnym `StratifiedGroupKFold` do grupowego doboru parametru `C`;
* **RF importance** — wykorzystuje ważność cech wyznaczoną przez Random Forest; wybierane są cechy o ważności większej od średniej.

Selekcja jest wykonywana wyłącznie na danych treningowych danego eksperymentu lub foldu. Jeżeli żadna cecha nie przekroczy progu, wybierana jest cecha z najwyższym wynikiem.

## Metryki

Wspólny moduł ewaluacji oblicza:

* accuracy;
* balanced accuracy;
* precision dla klasy anomalii;
* recall dla klasy anomalii;
* F1 dla klasy anomalii;
* specificity;
* false positive rate;
* ROC AUC;
* PR AUC;
* liczby TN, FP, FN i TP.

Pole `pr\\\\\\\_auc` jest w kodzie obliczane za pomocą `average\\\\\\\_precision\\\\\\\_score`, dlatego w opisie pracy należy określać je jako **Average Precision (AP)** lub wyraźnie zaznaczyć przyjęty sposób obliczenia PR AUC.

W części skryptów mierzone są również czasy treningu i predykcji.

## Struktura projektu

```text
Wykrywanie-anomalii-w-logach-aplikacji-webowej/
├── data/
│   └── csic\\\\\\\_database.csv
├── models/                         # zapisane modele i zbiory testowe
├── reports/                        # wyniki CSV i wykresy
├── artifacts/                      # miejsce na dodatkowe artefakty
├── splits/
│   └── group\\\\\\\_split\\\\\\\_seed\\\\\\\_42.csv
├── result tables/                  # archiwalne zestawienia wyników w PDF
├── sources/                        # publikacje i materiały źródłowe
├── cechy\\\\\\\_literaturowe.py
├── config.py
├── ewaluacja.py
├── metryki.py
├── model\\\\\\\_iforest.py
├── model\\\\\\\_iforest\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py
├── model\\\\\\\_ocsvm.py
├── model\\\\\\\_ocsvm\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py
├── model\\\\\\\_rand\\\\\\\_forest.py
├── model\\\\\\\_rand\\\\\\\_forest\\\\\\\_tfidf\\\\\\\_ngram.py
├── podzial\\\\\\\_danych.py
├── porownanie\\\\\\\_cech\\\\\\\_modeli\\\\\\\_jednoklasowych.py
├── porownanie\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py
├── porownanie\\\\\\\_podzialow\\\\\\\_random\\\\\\\_forest.py
├── porownanie\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py
├── przetwarzanie\\\\\\\_danych.py
├── przygotowanie\\\\\\\_podzialu.py
├── selekcja\\\\\\\_cech.py
├── strojenie\\\\\\\_modeli\\\\\\\_jednoklasowych.py
├── walidacja\\\\\\\_cech\\\\\\\_literaturowych.py
├── walidacja\\\\\\\_modeli\\\\\\\_jednoklasowych\\\\\\\_cv10.py
├── walidacja\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest\\\\\\\_cv10.py
├── requirements.txt
└── README.md
```

Katalogi `models`, `reports` i `artifacts` są tworzone automatycznie, gdy są potrzebne.

## Opis modułów

### `config.py`

Centralny plik konfiguracyjny projektu. Definiuje ścieżki do danych, modeli, raportów, artefaktów i zapisanego podziału grupowego. Zawiera nazwy wymaganych kolumn, zestawy cech BASIC, ADVANCED, ALTHUBITI\_9 i ALTHUBITI\_5, wzorce bezpieczeństwa, ziarno losowości, rozmiar testu oraz domyślne hiperparametry Random Forest, Isolation Forest i One-Class SVM.

Zmiany wspólnych parametrów eksperymentów należy w pierwszej kolejności wykonywać właśnie w tym pliku.

### `przetwarzanie\\\\\\\_danych.py`

Odpowiada za kompletny proces przygotowania danych:

* wczytuje `data/csic\\\\\\\_database.csv`;
* sprawdza obecność wymaganych kolumn;
* uzupełnia brakujące wartości tekstowe;
* tworzy podstawowe cechy długości i liczby znaków specjalnych;
* tworzy opcjonalne flagi bezpieczeństwa;
* buduje kolumnę `request\\\\\\\_text`;
* dodaje cechy literaturowe;
* koduje etykiety do formatu `0/1`.

Funkcja `get\\\\\\\_processed\\\\\\\_data()` jest głównym punktem wejścia używanym przez pozostałe moduły.

### `cechy\\\\\\\_literaturowe.py`

Implementuje ekstrakcję dziewięciu cech inspirowanych literaturą. Moduł oczyszcza URL z końcówki wersji HTTP, rozdziela ścieżkę i query string, łączy argumenty URL z treścią żądania, a następnie oblicza długości, liczby argumentów, liter, cyfr, znaków specjalnych oraz maksymalną wartość bajtu.

### `podzial\\\\\\\_danych.py`

Zawiera wspólną obsługę podziałów danych:

* tworzy skrót SHA-256 żądania;
* przypisuje rekordy do grup;
* tworzy i zapisuje podział grupowy;
* wczytuje zapisany podział i sprawdza jego zgodność z danymi;
* tworzy podział losowy;
* zwraca indeksy wybranego rodzaju podziału;
* oblicza liczbę wspólnych grup między treningiem i testem.

### `przygotowanie\\\\\\\_podzialu.py`

Skrypt uruchomieniowy tworzący odtwarzalny podział grupowy. Przetwarza dane, wywołuje funkcję podziału i zapisuje wynik w `splits/group\\\\\\\_split\\\\\\\_seed\\\\\\\_42.csv`. Należy go uruchomić ponownie po każdej zmianie danych wejściowych lub sposobu budowania `request\\\\\\\_text`.

### `metryki.py`

Udostępnia wspólną funkcję obliczania metryk klasyfikacji binarnej. Zapewnia identyczny sposób liczenia wyników we wszystkich eksperymentach. Zawiera również konwersję predykcji modeli jednoklasowych z formatu `1/-1` na etykiety projektu `0/1`.

### `selekcja\\\\\\\_cech.py`

Implementuje Information Gain, selekcję opartą na regularyzacji L1 oraz selekcję na podstawie ważności cech Random Forest. Zwraca listę wybranych cech i szczegółowe oceny potrzebne do analizy stabilności selekcji.

### `walidacja\\\\\\\_cech\\\\\\\_literaturowych.py`

Sprawdza poprawność zestawów ALTHUBITI\_9 i ALTHUBITI\_5. Kontroluje obecność cech, brak wartości pustych i ujemnych, wyświetla statystyki opisowe oraz potwierdza, że ALTHUBITI\_5 jest podzbiorem ALTHUBITI\_9.

### `model\\\\\\\_rand\\\\\\\_forest.py`

Bazowy model Random Forest wykorzystujący ręcznie utworzone cechy wskazane przez `config.ML\\\\\\\_FEATURES`. Stosuje historyczny, losowy podział 80/20, trenuje model na obu klasach, zapisuje model i zbiór testowy w katalogu `models` oraz wyświetla ważność cech.

### `model\\\\\\\_iforest.py`

Bazowy Isolation Forest na cechach liczbowych. Stosuje historyczny, losowy podział 80/20, ale model trenuje wyłącznie na normalnych rekordach części treningowej. Zapisuje model i zbiór testowy w katalogu `models`.

### `model\\\\\\\_ocsvm.py`

Bazowy One-Class SVM na cechach liczbowych. Stosuje historyczny, losowy podział 80/20. Pipeline zawiera `StandardScaler` i One-Class SVM, a uczenie odbywa się wyłącznie na normalnych rekordach części treningowej. Model i zbiór testowy są zapisywane w katalogu `models`.

### `ewaluacja.py`

Wczytuje trzy zapisane modele bazowe i odpowiadające im zbiory testowe. Generuje raporty klasyfikacji, wspólną tabelę metryk, macierze pomyłek oraz krzywe ROC.

Ten moduł dotyczy wyłącznie historycznego podziału losowego 80/20. Wygenerowanych przez niego wykresów nie należy przedstawiać jako wyników końcowego podziału grupowego.

### `porownanie\\\\\\\_podzialow\\\\\\\_random\\\\\\\_forest.py`

Uruchamia ten sam model Random Forest z zestawem BASIC na podziale losowym i grupowym 80/20. Porównuje wyniki oraz liczbę wspólnych grup. Eksperyment pokazuje, jak duplikaty żądań wpływają na ocenę modelu.

### `porownanie\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py`

Eksperyment eksploracyjny porównujący zestawy BASIC, ALTHUBITI\_9 i ALTHUBITI\_5 dla Random Forest. Korzysta z losowego podziału 60/40, oblicza metryki, czasy oraz ważność cech. Wyniki mają jawne oznaczenie `split=random`.

### `porownanie\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py`

Eksperyment eksploracyjny porównujący brak selekcji, literaturowy podzbiór pięciu cech, Information Gain, L1 i RF importance. Selekcja jest wykonywana na treningu, a ocena odbywa się na losowym podziale 60/40. Moduł zapisuje zarówno metryki, jak i szczegółowe oceny cech.

### `walidacja\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest\\\\\\\_cv10.py`

Końcowa walidacja wpływu selekcji cech dla Random Forest. Wykorzystuje 10-krotną `StratifiedGroupKFold`, wykonuje selekcję osobno w każdym foldzie i kontroluje brak nakładania się grup. Zapisuje wyniki poszczególnych foldów, średnie i odchylenia standardowe, oceny cech, stabilność selekcji oraz parametry L1.

### `porownanie\\\\\\\_cech\\\\\\\_modeli\\\\\\\_jednoklasowych.py`

Porównuje Isolation Forest i One-Class SVM na zestawach BASIC, ALTHUBITI\_9 oraz zestawach wybranych przez Information Gain i RF importance. Wykorzystuje grupowy podział 60/40. Selekcja odbywa się wyłącznie na treningu, a modele są uczone wyłącznie na normalnych rekordach.

### `strojenie\\\\\\\_modeli\\\\\\\_jednoklasowych.py`

Odpowiada za strojenie Isolation Forest i One-Class SVM. Tworzy rozłączne grupowo zbiory treningowy, walidacyjny i testowy. Selekcja cech oraz wybór hiperparametrów odbywają się bez używania zbioru testowego. Po wybraniu parametrów model jest ponownie uczony na zewnętrznym treningu i oceniany na końcowym teście.

Moduł zapisuje pełne wyniki przeszukiwania siatki oraz końcowe wyniki najlepszych konfiguracji.

### `walidacja\\\\\\\_modeli\\\\\\\_jednoklasowych\\\\\\\_cv10.py`

Wykonuje 10-krotną walidację grupową Isolation Forest i One-Class SVM. Porównuje konfiguracje domyślne z konfiguracjami dostrojonymi dla poszczególnych zestawów cech. Selekcja Information Gain i RF importance jest powtarzana na treningu każdego foldu. Moduł zapisuje wyniki foldów, zbiorcze średnie i odchylenia standardowe oraz stabilność wybranych zestawów cech.

Stałe `TUNED\\\\\\\_ISOLATION\\\\\\\_FOREST` i `TUNED\\\\\\\_ONE\\\\\\\_CLASS\\\\\\\_SVM` powinny odpowiadać wynikom ostatniego uruchomienia `strojenie\\\\\\\_modeli\\\\\\\_jednoklasowych.py`.

### `model\\\\\\\_rand\\\\\\\_forest\\\\\\\_tfidf\\\\\\\_ngram.py`

Trenuje nadzorowany Random Forest na tekstowej reprezentacji żądań HTTP. Pipeline wykorzystuje znakowy `TfidfVectorizer` i Random Forest. Obsługuje podział losowy lub grupowy, zapisuje parametry reprezentacji, metryki, liczebności zbiorów i `group\\\\\\\_overlap`.

### `model\\\\\\\_iforest\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py`

Trenuje Isolation Forest na tekstowej reprezentacji żądań. Pipeline obejmuje znakowe n-gramy TF-IDF, Truncated SVD, `StandardScaler` i Isolation Forest. Model jest uczony wyłącznie na normalnych żądaniach treningowych. Skrypt zapisuje parametry i komplet metryk.

### `model\\\\\\\_ocsvm\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py`

Trenuje One-Class SVM na tekstowej reprezentacji żądań. Pipeline obejmuje znakowe n-gramy TF-IDF, Truncated SVD, `StandardScaler` i One-Class SVM z jądrem RBF. Model jest uczony wyłącznie na normalnych żądaniach treningowych. Skrypt zapisuje parametry i komplet metryk.

## Pliki wynikowe

Wyniki są zapisywane w katalogu `reports`.

|Skrypt|Pliki wynikowe|
|-|-|
|`przygotowanie\\\\\\\_podzialu.py`|`splits/group\\\\\\\_split\\\\\\\_seed\\\\\\\_42.csv`|
|`model\\\\\\\_rand\\\\\\\_forest.py`|`models/rf\\\\\\\_model.pkl`, `models/test\\\\\\\_data\\\\\\\_rf.pkl`|
|`model\\\\\\\_iforest.py`|`models/iforest\\\\\\\_model.pkl`, `models/test\\\\\\\_data\\\\\\\_if.pkl`|
|`model\\\\\\\_ocsvm.py`|`models/ocsvm\\\\\\\_model.pkl`, `models/test\\\\\\\_data\\\\\\\_oc.pkl`|
|`ewaluacja.py`|`metrics\\\\\\\_summary\\\\\\\_random\\\\\\\_80\\\\\\\_20.csv`, `confusion\\\\\\\_matrices\\\\\\\_random\\\\\\\_80\\\\\\\_20.png`, `roc\\\\\\\_curves\\\\\\\_random\\\\\\\_80\\\\\\\_20.png`|
|`porownanie\\\\\\\_podzialow\\\\\\\_random\\\\\\\_forest.py`|`rf\\\\\\\_split\\\\\\\_comparison.csv`|
|`porownanie\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py`|`rf\\\\\\\_literature\\\\\\\_features\\\\\\\_random\\\\\\\_60\\\\\\\_40.csv`, `rf\\\\\\\_literature\\\\\\\_feature\\\\\\\_importances\\\\\\\_random.csv`|
|`porownanie\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py`|`rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_random\\\\\\\_60\\\\\\\_40.csv`, `rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_scores\\\\\\\_random.csv`|
|`walidacja\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest\\\\\\\_cv10.py`|`rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_group\\\\\\\_cv10\\\\\\\_folds.csv`, `rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_group\\\\\\\_cv10\\\\\\\_summary.csv`, `rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_group\\\\\\\_cv10\\\\\\\_scores.csv`, `rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_group\\\\\\\_cv10\\\\\\\_stability.csv`, `rf\\\\\\\_feature\\\\\\\_selection\\\\\\\_group\\\\\\\_cv10\\\\\\\_l1\\\\\\\_parameters.csv`|
|`porownanie\\\\\\\_cech\\\\\\\_modeli\\\\\\\_jednoklasowych.py`|`one\\\\\\\_class\\\\\\\_group\\\\\\\_literature\\\\\\\_features\\\\\\\_60\\\\\\\_40.csv`|
|`strojenie\\\\\\\_modeli\\\\\\\_jednoklasowych.py`|`one\\\\\\\_class\\\\\\\_group\\\\\\\_parameter\\\\\\\_search\\\\\\\_validation.csv`, `one\\\\\\\_class\\\\\\\_group\\\\\\\_tuned\\\\\\\_test\\\\\\\_60\\\\\\\_40.csv`|
|`walidacja\\\\\\\_modeli\\\\\\\_jednoklasowych\\\\\\\_cv10.py`|`one\\\\\\\_class\\\\\\\_group\\\\\\\_retuned\\\\\\\_cv10\\\\\\\_configs\\\\\\\_folds.csv`, `one\\\\\\\_class\\\\\\\_group\\\\\\\_retuned\\\\\\\_cv10\\\\\\\_configs\\\\\\\_summary.csv`, `one\\\\\\\_class\\\\\\\_group\\\\\\\_retuned\\\\\\\_cv10\\\\\\\_selection.csv`|
|`model\\\\\\\_rand\\\\\\\_forest\\\\\\\_tfidf\\\\\\\_ngram.py`|`random\\\\\\\_forest\\\\\\\_tfidf\\\\\\\_ngram\\\\\\\_group.csv` albo wariant `random`|
|`model\\\\\\\_iforest\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py`|`isolation\\\\\\\_forest\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd\\\\\\\_group.csv` albo wariant `random`|
|`model\\\\\\\_ocsvm\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py`|`one\\\\\\\_class\\\\\\\_svm\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd\\\\\\\_group.csv` albo wariant `random`|

## Wymagania

Projekt został przygotowany dla:

```text
Python 3.14.6
```

Główne zależności są zapisane w `requirements.txt`:

```text
numpy==2.5.1
pandas==3.0.5
scipy==1.18.0
scikit-learn==1.9.0
matplotlib==3.11.1
seaborn==0.13.2
joblib==1.5.3
```

## Instalacja

Repozytorium należy sklonować, a następnie otworzyć jego główny katalog w PyCharmie lub terminalu:

```bash
git clone https://github.com/rgraba/Wykrywanie-anomalii-w-logach-aplikacji-webowej.git
cd Wykrywanie-anomalii-w-logach-aplikacji-webowej
```

Utworzenie środowiska wirtualnego:

```bash
python -m venv .venv
```

Aktywacja w Windows PowerShell:

```powershell
.venv\\\\\\\\Scripts\\\\\\\\Activate.ps1
```

Instalacja zależności:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Wszystkie polecenia należy wykonywać z głównego katalogu projektu.

## Zalecana kolejność uruchamiania

### 1\. Kontrola przetwarzania danych

```bash
python przetwarzanie\\\\\\\_danych.py
python walidacja\\\\\\\_cech\\\\\\\_literaturowych.py
```

### 2\. Przygotowanie odtwarzalnego podziału grupowego

```bash
python przygotowanie\\\\\\\_podzialu.py
python porownanie\\\\\\\_podzialow\\\\\\\_random\\\\\\\_forest.py
```

### 3\. Odtworzenie historycznych modeli bazowych

Ten etap jest opcjonalny i dotyczy podziału losowego 80/20:

```bash
python model\\\\\\\_rand\\\\\\\_forest.py
python model\\\\\\\_iforest.py
python model\\\\\\\_ocsvm.py
python ewaluacja.py
```

### 4\. Eksperymenty eksploracyjne dotyczące cech

Te skrypty wykorzystują podział losowy 60/40 i służą pokazaniu rozwoju metodologii:

```bash
python porownanie\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py
python porownanie\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest.py
```

### 5\. Końcowa walidacja selekcji cech Random Forest

```bash
python walidacja\\\\\\\_selekcji\\\\\\\_cech\\\\\\\_random\\\\\\\_forest\\\\\\\_cv10.py
```

### 6\. Analiza i strojenie modeli jednoklasowych

```bash
python porownanie\\\\\\\_cech\\\\\\\_modeli\\\\\\\_jednoklasowych.py
python strojenie\\\\\\\_modeli\\\\\\\_jednoklasowych.py
```

Po strojeniu należy sprawdzić, czy parametry zapisane w `TUNED\\\\\\\_ISOLATION\\\\\\\_FOREST` i `TUNED\\\\\\\_ONE\\\\\\\_CLASS\\\\\\\_SVM` odpowiadają najlepszym konfiguracjom z aktualnego przebiegu. Następnie można uruchomić:

```bash
python walidacja\\\\\\\_modeli\\\\\\\_jednoklasowych\\\\\\\_cv10.py
```

### 7\. Końcowe eksperymenty z reprezentacją tekstową

Przed uruchomieniem należy sprawdzić parametry końcowego wywołania znajdującego się na dole każdego pliku. Następnie należy wykonać:

```bash
python model\\\\\\\_rand\\\\\\\_forest\\\\\\\_tfidf\\\\\\\_ngram.py
python model\\\\\\\_iforest\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py
python model\\\\\\\_ocsvm\\\\\\\_tfidf\\\\\\\_2gram\\\\\\\_svd.py
```

Skrypty końcowe powinny korzystać z:

```text
split\\\\\\\_type="group"
```

## Odtwarzalność eksperymentów

* wspólne ziarno losowości: `42`;
* zależności przypięte do konkretnych wersji;
* zapisany podział grupowy;
* kontrola `group\\\\\\\_overlap`;
* selekcja cech wykonywana wyłącznie na treningu;
* skalowanie dopasowywane wyłącznie do danych treningowych;
* modele jednoklasowe uczone wyłącznie na próbkach normalnych;
* wyniki zapisują rodzaj podziału oraz użyte parametry.

Poprawność składni wszystkich modułów można dodatkowo sprawdzić poleceniem:

```bash
python -m compileall -q .
```

## Interpretacja wyników

Za podstawę końcowych wniosków należy przyjmować eksperymenty wykorzystujące podział grupowy. Wyniki z podziałów losowych pełnią funkcję bazową i eksploracyjną oraz powinny być jednoznacznie podpisane w pracy.

W przypadku danych niezbalansowanych głównymi metrykami porównawczymi są:

* balanced accuracy;
* F1 dla klasy anomalii;
* ROC AUC;
* Average Precision;
* false positive rate.

Sama accuracy nie powinna stanowić jedynego kryterium wyboru modelu.

## Źródła i materiały porównawcze

Publikacje wykorzystane podczas projektowania cech i porównywania wyników znajdują się w katalogu:

```text
sources/
```

Archiwalne zestawienia wyników znajdują się w katalogu:

```text
result tables/
```

Materiały te służą do udokumentowania podstaw literaturowych, wcześniejszych etapów badań oraz porównania rezultatów projektu z innymi pracami wykorzystującymi zbiór CSIC 2010.

