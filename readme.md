# Wykrywanie anomalii w logach aplikacji webowej

Projekt stanowi część praktyczną pracy magisterskiej dotyczącej wykrywania anomalii i potencjalnych ataków na podstawie żądań HTTP ze zbioru CSIC 2010.

Głównym celem projektu nie jest wyłącznie porównanie Random Forest, Isolation Forest i One-Class SVM. Najważniejszym zagadnieniem badawczym jest wpływ duplikatów żądań HTTP oraz sposobu podziału danych na ocenę skuteczności modeli.

## Pytania badawcze

Projekt odpowiada na trzy główne pytania:

1. **RQ1.** W jakim stopniu losowy podział rekordów zawyża wyniki wykrywania anomalii na zbiorze CSIC 2010 w porównaniu z podziałem uwzględniającym duplikaty?
2. **RQ2.** Które modele i reprezentacje cech zachowują dobrą skuteczność po zastosowaniu bardziej rygorystycznego, grupowego podziału danych?
3. **RQ3.** Jak zmieniają się wnioski dotyczące skuteczności Random Forest, Isolation Forest i One-Class SVM po wyeliminowaniu przecieku wynikającego z powtarzających się żądań?

## Zbiór danych

W projekcie wykorzystano zbiór **CSIC 2010 Web Application Attacks**. Plik wejściowy powinien znajdować się pod ścieżką:

```text
data/csic_database.csv
```

Wymagane kolumny:

- `Method` — metoda HTTP;
- `URL` — adres lub ścieżka żądania;
- `content` — treść żądania;
- `classification` — etykieta klasy.

Problem został sprowadzony do klasyfikacji binarnej:

- `0` — ruch normalny;
- `1` — anomalia lub potencjalny atak.

Zbiór zawiera 61 065 rekordów, w tym 36 000 rekordów normalnych i 25 065 anomalii.

## Duplikaty i identyfikator grupy

Grupą jest identyczne żądanie HTTP. Identyfikator `request_group` jest wyznaczany jako skrót SHA-256 jednoznacznej serializacji pól:

```text
[Method, URL, content]
```

Audyt danych wykazał:

- 25 608 unikalnych grup żądań;
- 35 457 rekordów będących dodatkowymi wystąpieniami duplikatów;
- 37 361 rekordów należących do grup zawierających duplikaty;
- 1 904 grupy duplikatów;
- maksymalny rozmiar grupy równy 1000;
- brak grup zawierających jednocześnie obie etykiety;
- brak wykrytych kolizji skrótów.

Kolumna `request_text`, wykorzystywana przez reprezentacje tekstowe, powstaje z połączenia metody HTTP, adresu URL i treści żądania. Nie pełni ona roli indeksu rekordu ani identyfikatora podziału.

## Protokół eksperymentalny

Podstawą wszystkich końcowych eksperymentów jest jeden zapisany protokół grupowy. Znajduje się on w plikach:

```text
splits/experimental_protocol_seed_2026.csv
splits/experimental_protocol_seed_2026.json
```

Plik CSV zawiera przypisanie każdego rekordu do części `development` albo `final_test` oraz numer foldu walidacyjnego dla rekordów `development`. Plik JSON zawiera parametry protokołu, statystyki grup i odcisk danych wejściowych.

### Podział development/final_test

Podział jest wykonywany przez `StratifiedGroupKFold` z `PROTOCOL_RANDOM_STATE = 2026`:

- `development`: 48 852 rekordy, czyli 80% danych;
- `final_test`: 12 213 rekordów, czyli 20% danych.

Grupy identycznych żądań są niepodzielne. Pomiędzy `development` i `final_test` musi zachodzić:

```text
group_overlap = 0
```

Zbiór `final_test` nie jest używany do wyboru modelu, cech, hiperparametrów ani progu decyzyjnego.

### Walidacja na development

W obrębie `development` stosowana jest 10-krotna `StratifiedGroupKFold`. Każdy rekord występuje dokładnie raz w walidacji OOF, a dla każdego foldu kod sprawdza:

```text
group_overlap = 0
```

Wszystkie porównywane modele korzystają z tych samych zapisanych foldów.

### Kontrolowany podział losowy

Zwykły `StratifiedKFold` występuje wyłącznie w `porownanie_podzialow_random_forest.py`. Jest to celowy wariant kontrolny dla RQ1, a nie końcowy protokół oceny.

Eksperyment porównuje na całym zbiorze `development`:

- losowy podział rekordów `random_record`, w którym mierzone jest nakładanie grup;
- podział `group`, w którym nakładanie grup musi wynosić zero.

Zbiór `final_test` nie jest wykorzystywany w tym porównaniu.

## Automatyczne zabezpieczenia przed przeciekiem

Funkcja `ensure_no_group_overlap()` jest wywoływana przez każdy skrypt korzystający z grupowych foldów. W przypadku wykrycia wspólnej grupy przerywa eksperyment wyjątkiem.

Testy w `tests/test_protokol_eksperymentalny.py` sprawdzają między innymi:

- brak wspólnych grup między `development` i `final_test`;
- brak wspólnych rekordów i grup w każdym foldzie;
- pełne i jednokrotne pokrycie `development` przez foldy walidacyjne;
- zakaz używania wycofanych mechanizmów podziału;
- ograniczenie zwykłego `StratifiedKFold` do eksperymentu RQ1;
- obowiązkową kontrolę `group_overlap` przez skrypty korzystające z foldów;
- ograniczenie dostępu do indeksów `final_test` do protokołu i końcowej ewaluacji.

## Modele i scenariusze uczenia

W projekcie wykorzystywane są trzy modele:

- **Random Forest** — klasyfikator nadzorowany uczony na rekordach normalnych i anomalnych;
- **Isolation Forest** — model wykrywania anomalii uczony wyłącznie na rekordach normalnych;
- **One-Class SVM** — model jednoklasowy uczony wyłącznie na rekordach normalnych.

Modele jednoklasowe są analizowane w dwóch oddzielnych scenariuszach.

### Supervised-calibrated one-class

Sam model jest nadal dopasowywany wyłącznie do normalnych rekordów treningowych, ale etykiety anomalii ze zbioru `development` mogą uczestniczyć w porównaniu konfiguracji i wyborze wariantu. Wyniki tego scenariusza opisują model korzystający z dodatkowej informacji nadzorowanej na etapie wyboru.

### Strict one-class

Uczenie oraz kalibracja progu wykorzystują wyłącznie normalne rekordy treningowe. Normalne dane są dzielone grupowo na część dopasowania i kalibracji. Anomalie nie są używane do wyboru parametrów ani progu.

Końcowy próg strict one-class jest kalibrowany do docelowego FPR równego 5%. Niezależnie od tego raportowane są również punkty pracy TPR przy FPR równym 1%, 5% i 10%.

## Reprezentacje i zestawy cech

### BASIC

Podstawowy zestaw czterech cech liczbowych:

- `url_len`;
- `content_len`;
- `url_special_chars`;
- `content_special_chars`.

### ALTHUBITI_9

Zestaw dziewięciu cech odtworzonych na podstawie literatury:

- `request_length`;
- `arguments_length`;
- `arguments_count`;
- `arguments_digit_count`;
- `path_length`;
- `arguments_letter_count`;
- `path_letter_count`;
- `path_special_char_count`;
- `max_request_byte`.

### ALTHUBITI_5

Pięcioelementowy podzbiór cech literaturowych:

- `request_length`;
- `arguments_length`;
- `arguments_count`;
- `path_length`;
- `path_special_char_count`.

### ADVANCED i reprezentacje tekstowe

Zestaw ADVANCED rozszerza BASIC o flagi SQL Injection, XSS i Path Traversal. Repozytorium zawiera również eksperymenty TF-IDF oparte na znakowych n-gramach, a w modelach jednoklasowych dodatkowo na Truncated SVD i standaryzacji.

Są to eksperymenty uzupełniające. Nie zastępują głównego protokołu grupowego i nie powinny być przedstawiane jako wyniki końcowe, jeżeli korzystają z historycznego podziału rekordów.

## Selekcja cech

Projekt implementuje:

- **Information Gain** — wybór na podstawie informacji wzajemnej;
- **L1/LASSO** — pipeline `StandardScaler` i `LogisticRegression` z regularyzacją L1;
- **RF importance** — wybór na podstawie ważności cech Random Forest;
- stałe zestawy BASIC, ALTHUBITI_5 i ALTHUBITI_9.

Selekcja jest wykonywana osobno na części treningowej każdego zewnętrznego foldu. Skalowanie jest dopasowywane wyłącznie na danych treningowych. Dobór parametru `C` dla L1 odbywa się w wewnętrznej, 5-krotnej `StratifiedGroupKFold`, do której przekazywane są grupy żądań.

## Metryki i niepewność

Wspólny moduł `metryki.py` oblicza:

- accuracy;
- balanced accuracy;
- precision, recall i F1 dla klasy anomalii;
- specificity i false positive rate;
- ROC AUC;
- Average Precision;
- TPR przy FPR równym 1%, 5% i 10%;
- liczby TN, FP, FN i TP.

Pole `pr_auc` jest obliczane funkcją `average_precision_score`, dlatego w pracy należy nazywać je **Average Precision (AP)** albo wyraźnie podać przyjętą definicję.

Wyniki 10-krotnej walidacji są raportowane jako średnia i odchylenie standardowe. Dla głównych różnic między metodami wyznaczane są 95-procentowe przedziały ufności za pomocą sparowanego bootstrapu grupowego:

- jednostką losowania jest cała grupa identycznych żądań;
- porównywane metody korzystają z tej samej próby bootstrapowej;
- domyślna liczba iteracji wynosi 2000;
- różnice są liczone dla tych samych rekordów i foldów.

Wyniki wariantów dobieranych na podstawie tych samych foldów `development` są wynikami etapu wyboru modelu, a nie niezależną oceną końcową. Podstawą oceny potwierdzającej pozostaje zamrożony `final_test`.

## Zamrożone konfiguracje końcowe

Po zakończeniu analiz na `development` konfiguracje końcowe zostały zapisane jawnie w `config.py`:

- Random Forest: zestaw `INFORMATION_GAIN` z cechami `request_length`, `arguments_length` i `arguments_letter_count`;
- Isolation Forest, supervised-calibrated: `ALTHUBITI_9`;
- One-Class SVM, supervised-calibrated: `BASIC`;
- Isolation Forest, strict: `BASIC`, próg kalibrowany na normalnych rekordach do FPR 5%;
- One-Class SVM, strict: `BASIC`, próg kalibrowany na normalnych rekordach do FPR 5%.

Stałe `FINAL_*` są celowo ręcznie zamrożonym zapisem decyzji podjętych przed otwarciem `final_test`. Nie należy ich automatycznie aktualizować po wykonaniu końcowej ewaluacji. Jeżeli dane, protokół lub finalne konfiguracje zostaną później zmienione, będzie to nowy eksperyment, a nie kontynuacja tej samej oceny końcowej.

Konfiguracje określane jako `TUNED` są wybierane na `development`. Ich wyniki walidacyjne mogą zawierać niewielki optymizm selekcyjny, ponieważ wybór konfiguracji i jej podsumowanie korzystają z tych samych 10 foldów. Nie narusza to niezależności zamrożonego `final_test`, ale wyniki `TUNED` należy opisywać jako etap wyboru modelu, a nie jako niezależny wynik końcowy.

## Najważniejsze moduły

- `config.py` — ścieżki, ziarna, parametry modeli, zestawy cech i zamrożone konfiguracje końcowe;
- `przetwarzanie_danych.py` — wczytywanie, czyszczenie i ekstrakcja cech;
- `podzial_danych.py` — tworzenie grup żądań i kontrola nakładania grup;
- `protokol_eksperymentalny.py` — zapis i odczyt podziału `development`/`final_test` oraz 10 foldów grupowych;
- `metryki.py` — wspólna implementacja metryk;
- `selekcja_cech.py` — Information Gain, L1 i RF importance;
- `porownanie_podzialow_random_forest.py` — kontrolowane porównanie `random_record` z `group` dla RQ1;
- `analiza_niepewnosci.py` — grupowy bootstrap różnicy między podziałami;
- `walidacja_selekcji_cech_random_forest_cv10.py` — grupowa walidacja zestawów i selekcji cech Random Forest;
- `walidacja_modeli_jednoklasowych_cv10.py` — grupowa walidacja modeli jednoklasowych w scenariuszu strict i supervised-calibrated;
- `porownanie_modeli_cv10.py` — sparowane porównanie modeli na predykcjach OOF i grupowy bootstrap;
- `ewaluacja.py` — test implementacji na `development` oraz jednorazowa ewaluacja `final_test`;
- `tests/test_protokol_eksperymentalny.py` — testy protokołu i bezpiecznego użycia podziałów.

Plik `ewaluacja_80_20.py` oraz część skryptów `model_*.py` i starszych skryptów porównawczych służą do odtworzenia etapów historycznych lub eksperymentów uzupełniających. Nie są główną ścieżką końcowej metodologii.

## Pliki wynikowe

Wyniki są zapisywane w katalogu `reports/`, który jest ignorowany przez Git. Dzięki temu ponowne uruchomienie eksperymentów nie zaśmieca historii repozytorium. Ważne wyniki należy zachować poza katalogiem roboczym i wykorzystać podczas tworzenia końcowego raportu.

### RQ1: podział losowy i grupowy

```text
reports/rf_random_vs_group_cv10_folds.csv
reports/rf_random_vs_group_cv10_summary.csv
reports/rf_random_vs_group_cv10_oof_predictions.csv
reports/rf_random_vs_group_group_bootstrap_samples.csv
reports/rf_random_vs_group_group_bootstrap_summary.csv
```

### Selekcja cech Random Forest

```text
reports/rf_feature_selection_group_cv10_folds.csv
reports/rf_feature_selection_group_cv10_summary.csv
reports/rf_feature_selection_group_cv10_scores.csv
reports/rf_feature_selection_group_cv10_stability.csv
reports/rf_feature_selection_group_cv10_l1_parameters.csv
```

### Modele jednoklasowe

```text
reports/one_class_group_cv10_configs_folds.csv
reports/one_class_group_cv10_configs_summary.csv
reports/one_class_group_cv10_selection.csv
reports/one_class_group_cv10_oof_predictions.csv.gz
```

### Porównanie modeli

```text
reports/model_comparison_group_cv10_bootstrap_samples.csv.gz
reports/model_comparison_group_cv10_bootstrap_model_summary.csv
reports/model_comparison_group_cv10_bootstrap_pairwise_summary.csv
```

### Jednorazowa ewaluacja final_test

```text
reports/final_test_metrics_seed_2026.csv
reports/final_test_predictions_seed_2026.csv.gz
```

Tabela wyników do pracy i raport dla prowadzącego powinny zostać utworzone dopiero po ponownym wygenerowaniu i kontroli wszystkich wyników `development`. Istniejących plików `final_test` nie należy przy tym nadpisywać ani ponownie generować.

## Wymagania i instalacja

Projekt został przygotowany dla Pythona 3.14.6. Zależności są przypięte w `requirements.txt`.

```bash
git clone https://github.com/rgraba/Wykrywanie-anomalii-w-logach-aplikacji-webowej.git
cd Wykrywanie-anomalii-w-logach-aplikacji-webowej
git switch wersja-poprawiona
python -m venv .venv
```

Aktywacja środowiska w Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalacja zależności:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Wszystkie polecenia należy wykonywać z głównego katalogu projektu.

## Zalecana kolejność uruchamiania

### 1. Kontrola składni i protokołu

```bash
python -m compileall -q .
python -m unittest tests.test_protokol_eksperymentalny -v
```

Oczekiwanym wynikiem drugiego polecenia jest osiem zaliczonych testów.

Zapisany protokół należy traktować jako część zamrożonego eksperymentu. Nie należy go generować ponownie podczas zwykłego odtwarzania wyników. Nowy protokół tworzy się tylko dla świadomie rozpoczętego nowego eksperymentu, na przykład po zmianie danych lub definicji grupy.

### 2. Ponowne wygenerowanie wyników development

Polecenia należy uruchamiać w następującej kolejności:

```bash
python porownanie_podzialow_random_forest.py
python analiza_niepewnosci.py
python walidacja_selekcji_cech_random_forest_cv10.py
python walidacja_modeli_jednoklasowych_cv10.py
python porownanie_modeli_cv10.py
```

Każdy skrypt powinien potwierdzić, że `final_test` nie został użyty. `analiza_niepewnosci.py` wymaga predykcji OOF wygenerowanych przez porównanie podziałów, a `porownanie_modeli_cv10.py` wymaga predykcji OOF Random Forest i modeli jednoklasowych.

### 3. Test implementacji końcowej

```bash
python ewaluacja.py --smoke-test
```

Polecenie wykorzystuje pierwszy fold `development`, nie zapisuje wyników końcowych i nie otwiera `final_test`.

### 4. Jednorazowa ewaluacja final_test

```bash
python ewaluacja.py --run-final-test
```

To polecenie wolno wykonać dopiero po zamrożeniu wszystkich decyzji podjętych na `development`. Skrypt odmawia uruchomienia, jeżeli istnieje już którykolwiek z końcowych plików wynikowych.

W bieżącym eksperymencie ocena `final_test` została już wykonana. Podczas końcowego odtwarzania analiz należy uruchomić ponownie wyłącznie eksperymenty `development` i wykorzystać zachowane pliki końcowe, bez ponownego wywoływania `--run-final-test`.

### 5. Końcowy raport

Po poprawnym przejściu testów i ponownym wygenerowaniu wyników `development` należy:

1. sprawdzić kompletność i zgodność plików CSV;
2. połączyć wyniki `development` z zachowanymi wynikami `final_test`;
3. przygotować nowe tabele, wykresy i raport dla prowadzącego;
4. jasno rozdzielić wyniki wyboru modelu od jednorazowej oceny końcowej.

## Odtwarzalność i ograniczenia

- `PROTOCOL_RANDOM_STATE = 2026` odpowiada wyłącznie za zapisany protokół podziału;
- `RANDOM_STATE = 42` jest zachowany dla modeli i pomocniczych operacji losowych;
- odcisk danych w pliku JSON wykrywa zmianę kolejności, grup lub etykiet rekordów;
- wszystkie końcowe foldy są grupowe i mają `group_overlap = 0`;
- selekcja cech, skalowanie i strojenie odbywają się bez użycia `final_test`;
- modele jednoklasowe są dopasowywane wyłącznie do normalnych rekordów;
- bootstrap losuje całe grupy żądań, a nie pojedyncze rekordy;
- grupowanie usuwa przeciek dokładnych duplikatów, ale nie grupuje automatycznie żądań jedynie podobnych semantycznie.

## Interpretacja wyników

Podstawą głównych wniosków są eksperymenty z podziałem grupowym i jednorazowa ocena `final_test`. Wyniki podziału losowego pełnią rolę kontrolną w RQ1 i pokazują wpływ przecieku duplikatów.

W przypadku niezbalansowanych danych najważniejsze są balanced accuracy, F1 dla anomalii, Average Precision, ROC AUC, FPR oraz TPR przy ustalonych wartościach FPR. Sama accuracy nie powinna być używana jako jedyne kryterium wyboru.

Random Forest i modele jednoklasowe działają w różnych warunkach informacyjnych. Porównanie ich wyników musi zawsze wskazywać, czy dotyczy scenariusza strict one-class, supervised-calibrated one-class czy klasyfikacji nadzorowanej.

## Źródła

Publikacje wykorzystane przy projektowaniu cech i eksperymentów znajdują się w katalogu `sources/`. Archiwalne materiały i wcześniejsze zestawienia wyników znajdują się w katalogu `result tables/` i nie powinny być mylone z wynikami wygenerowanymi przez aktualny protokół seed 2026.
