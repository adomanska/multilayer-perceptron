# Multilayer Perceptron

Implementacja sieci neuronowej Percepron Wielowarstwowy w języku Python.

## Wymagania
W celu uruchomienia programu należy zainstalować:
- Python (co najmniej wersja 3.6.5)
- menedżer pakietów `pip`
- pakiet `virtualenv`

## Konfiguracja wirtualnego środowiska i instalacja pakietów
Wybrane przydatne polecenia (wykonywane w katalogu */backend*):

1. Stworzenie wirtualnego środowiska:
```
$ python3 -m venv .env
```

2. Aktywacja środowiska (POSIX):
```
$ source .env/bin/activate
```
3. Aktywacja śroowiska (Windows):
```
$ .env\Scripts\activate
```

4. Instalacja pakietów z pliku *requirements.txt*:
```
$ pip install -r requirements.txt --user
```

5. Instalacja nowych pakietów powinna być wykonywana po aktywowaniu środowiska, a następnie plik *requirements.txt* powininen być zaktualizowany:
```
$ pip freeze > requirements.txt
```

6. Deaktywacja wirtualnego środowiska:
```
$ deactivate
```

## Instrukcja uruchomienia
Aby uruchomić aplikację, należy z poziomu głównego katalogu projektu wykonać polecenie:
```
$ ./backend/web_application.py
```
