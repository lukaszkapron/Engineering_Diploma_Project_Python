Projekt stworzony na potrzeby pracy dyplowej


Temat pracy: "Wykorzystanie uczenia mszynowego do przewidywania wyników meczów piłkarskich"

Data obrony: 07.02.2024r.

Streszczenie: "Niniejsza praca zawiera proces budowy modelu predykcyjnego opartego o algorytmy uczenia
maszynowego w celu predykcji wyników meczów piłkarskich. Początek pracy jest wprowadzeniem do
badanego problemu oraz przeglądem literatury odnoszącej się do podobnej tematyki. Następnie na
podstawie postawionego celu, pozyskano odpowiednie dane o meczach piłkarskich i przystąpiono do
budowania skutecznego, wieloklasowego modelu predykcyjnego. Pierwszym etapem było odpowiednie
wyliczenie statystyk meczowych, mogących mieć największy wpływ na końcowy rezultat spotkania oraz
uzupełnienie braków danych. Następnie wykorzystując wizualizacje danych oraz metody selekcji
zmiennych wybrano te, które będą mogły przynieść najlepsze rezultaty. Na podstawie tak
przygotowanych zmiennych wytrenowano 3 algorytmy uczenia maszynowego, a następnie przetestowano
zbudowane modele na zbiorze danych testowych, starając się przy tym osiągnąć jak najbardziej
jakościowe wyniki. Na koniec omówiono uzyskane rezultaty oraz porównano je do innych, tego typu
systemów."


Projekt zawiera implementację trenowania algorytmów uczenia maszynowego, a następnie testowanie w celu
predykcji wyników meczów piłkarskich Premier League sezonu 2022/23. Celem pracy było przewidywanie wyników
(zwycięstwo drużyny domowej, remis, zwycięstwo drużyny przyjezdnej) z jak największą dokładnością.
Pozyskane dane zostały poddane procesowania badania współczynnik korelacji Pearsona, na podstawie
którego została stworzona macierz korelacji, a najbardziej skorelowane ze sobą zmienne zostały odrzucone
ze zbioru trenującego, wykorzystując przy tym technikę Mutual Information. Wytrenowane modele zostały 
przetestowane na zbiorze 380 meczy. Skuteczność jaką udało się osiągąć oscylowała w granicach 51-55%,
co jesty wynikiem dobrym, ze względu na klasyfikację wieloklasową).

Użyte technologie:
- Python
- numpy
- scikit-learn
- pandas
- matplotlib
- seaborn

Uwaga: zmienne użyte w procesie predykcji zostały przygotowane na podstawie zbioru danych dostępnego pod adresem:
- https://www.football-data.co.uk/englandm.php

Dane zostały przekształcone przeze mnie używając język programowania c#. Projekt jest dostępny pod adresem:
- https://github.com/lukaszkapron/Engineering_Diploma_Project_Csharp
