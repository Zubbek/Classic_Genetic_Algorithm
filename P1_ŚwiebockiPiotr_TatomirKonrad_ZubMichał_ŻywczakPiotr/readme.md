# Classic Genetic Algorithm

Aplikacja implementująca algorytm genetyczny celem rozwiązywania problemu optymalizacji (maksymalizacji oraz minimalizacji funkcji wielu zmiennych). Implementacja powinna
dać możliwość wyboru dowolnej liczby zmiennych w konfiguracji (to znaczy powinna być możliwość przetestowania algorytmu np. dla funkcji 5, 10, 20 czy 27 zmiennych).

## Zawartość projektu:

1. Implementacja binarnej reprezentacji chromosomu + konfiguracja dokładności
2. Implementacja konfiguracji wielkości populacji
3. Implementacja konfiguracji liczby epok
4. Implementacja metod selekcji najlepszych, kołem ruletki, selekcji turniejowej + konfiguracje parametrów
5. Implementacja krzyżowania jednopunktowego, dwupunktowego, krzyżowania jednorodnego, krzyżowania ziarnistego + konfiguracja prawdopodobieństwa krzyżowania.
6. Implementacji mutacji brzegowej, jedno oraz dwupunktowej + konfiguracja prawdopodobieństwa mutacji
7. Implementacja operatora inwersji + konfiguracja prawdopodobieństwa jego wystąpienia
8. Implementacja strategii elitarnej + konfiguracja % lub liczby osobników przechodzącej do kolejnej populacji 

## Wymagania środowiska

Do uruchomienia aplikacji wymagane jest zainstalowanie języka Python oraz następujących bibliotek:

- tkinter
- matplotlib
- numpy
- time
- benchmark_functions
- opfunu.cec_based.cec2014

## Instalacja

Aby zainstalować aplikację, wykonaj poniższe kroki:

```bash
git clone https://github.com/Zubbek/Classic_Genetic_Algorithm.git

cd Classic_Genetic_Algorithm

pip install -r requirements.txt
```
## Uruchomienie

Aby uruchomić aplikację, użyj poniższego polecenia:

```bash
python ./Gui.py
```