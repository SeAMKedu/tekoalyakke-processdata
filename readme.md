[seamk_logo]:       /img/Seamk_logo.svg
[epliitto_logo]:    /img/EPLiitto_logo_vaaka_vari.jpg

[real_results]:     /img/just_real_training_result_lock.png
[gan_results]:      /img/gan_training_result_lock.png
[model]:            /img/model.png 
[analysis]:         /img/sample_analysis.png

# Prosessidatan käsittelyä

Tässä Tekoäly-AKKE hankkeen demossa käymme läpi prosessidatan luokittelua ja analysointia. Data on peräisin oikeasta tehdasympäristöstä, mutta se on tähän liitetty mukaan hieman muokattuna ja anonymisoituna. Ensimmäisessä esimerkissä tehdään erilaisia analyysejä datalle ja koitetaan löytää sieltä pareja, jotka vaikuttavat toisiinsa. Toinen esimerkki luokittelee mittaustulosten perusteella, onko prosessin tulos minkälaatuinen.

## Luo ajoympäristö

Demossa käytetyt kirjastot ovat tätä tehdessä hieman ristiriidassa keskenään. Tensorflow ja sdv (GAN kirjasto) eivät uusimmilla versioilla tule keskenään toimeen tätä kirjoitettaessa. Jos aiot ajaa GAN koodia, voi olla helpompaa tehdä kaksi erillistä virtuaaliympäristöä ja olla asentamatta tensorflow:ta toiseen. Pythonin version ei myöskään tulisi olla yli 3.9.x. 

```
python -m venv venv-notf
source venv-notf/bin/activate
pip install --upgrade pip wheel
pip install --upgrade ctgan pandas numpy
deactivate

python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install --upgrade tensorflow pandas numpy matplotlib scikit-learn pydot
```

Tensorflown asennus M1 macciin menee tämän tapaisesti, katso ohjeet https://developer.apple.com/metal/tensorflow-plugin/
```
source ~/miniforge3/bin/activate
conda install -c apple tensorflow-deps==2.6.0
python -m venv venv
pip install --upgrade pip wheel
pip install --upgrade tensorflow-macos tensorflow-metal pandas numpy matplotlib scikit-learn pydot
```

## Esimerkki 1: Analyysi

Vaikka rivejä ei datasetissä ole kovin paljoa (vajaa 300), on mitattuja pisteitä per ajo yli 30. Ensi yhdistely ja siivoaminen tehtiin Knime:llä (https://www.knime.com/). Setistä pudotettiin ensin vakio arvot pois ja kellon aikojen perusteella laskettiin kestoja varten valmiiksi uudet sarakkeet. 

Tämän jälkeen etsimme mahdollisia ryhmiä katsomalla läpi arvojen pareja. K-means klusterointi-algoritmilla klusterin lukumäärää voidaan arvioida heuristisesti elbow-menetelmällä, jossa pyritään löytämään datasta kiintopisteitä ja niiden lukumääriä. Valitsemalla sopiva klusterimäärä ja asettamalla pisteitä, varianssi laskee merkittävästi ko. pisteiden ympärillä. Ohjelma pyrkii löytämään dataan parhaiten sopivan klusterimäärän, jota voi tarkastella elbow-menetelmän ja "Data & clusters" kuvaajista.

Confidence ellipsien muotoja tutkimalla etsimme pareja joilla olisi merkittävä vaikutus toisiinsa. Mitä kapeampi ellipsi on, sitä voimakkaamin arvot näyttävät vaikuttavan toisiinsa.

![analysis]

Ajamalla `analysis.py` ohjelman, se käy läpi datasetin ja kirjoittaa pdf-hakemistoon `analysis.pdf` tiedoston, mistä tuloksia voi käydä katselemassa.

```
python analysis.py
```

## Esimerkki 2: Luokittelu

Yllämainittujen versio ristiriitojen vuoksi, datan generointi erotettiin omaksi ohjelmakseen. Kun ajat tämän, se luo 10000 riviä alkuperäisen näköistä dataa ja tallentaa sen `data/gan.csv` tiedostoon.

```
source venv-notf/bin/activate
python generator.py
deactivate
```

Tämän jälkeen voit ajaa luokittelun pääympäristössä. 

```
source venv/bin/activate
python classification.py
```

Luokittelu tehdään kahteen kertaan, ensin pienempää datasettiä käyttäen ja toisen kerran isompaa synteettistä dataa käyttäen. 
Molempien ajojen lopuksi, tulee pieni taulukko, jossa opettu malli yrittää määritellä oliko kyseinen prosessin ajo lopputuloksen kannalta huono, ok, hyvä vai erinomainen.

```
Value   Label  Prediction Correct
1001.1  Good   Good       True
987.4   Good   Good       True
1186.0  Oh my  Oh my      True
1016.6  Good   Good       True
744.4   Bad    Bad        True
978.3   Good   Good       True
1007.3  Good   Good       True
1102.2  Oh my  Oh my      True
872.9   Bad    Ok         False
1039.3  Good   Good       True
```

Pienemmän datasetin käyttö näkyy ylisovittuminen. Sahaus kertoo myös vaikeuksista mallin validoinnissa. Pieni määrä yhdistettynä mittausten samankaltaisuuteen tuo myös toisen ongelman esiin, suurin osa tuloksista kuuluu yhteen ja samaan luokkaan. Kolme muuta luokkaa saavat vain 10% kaikista riveistä jaettavaksi. 

![real_results]

Kun saadaan käyttöön enemmän dataa, näyttävät käyrät paremmilta. 

![gan_results]

Opettaminen lopetetaan kun alkaa näyttää siltä, ettei parannusta enää tapahdu. Mallissa on myös yksi dropout kerros, joka yrittää pienentää ylisovittamista.

![model]


## Tekoäly-AKKE hanke

Syksystä 2021 syksyyn 2022 kestävässä hankkeessa selvitetään Etelä-Pohjanmaan alueen yritysten tietoja ja tarpeita tekoälyn käytöstä sekä tuodaan esille tapoja sen käyttämiseen eri tapauksissa, innostaen laajempaa käyttöä tällä uudelle teknologialle alueella. Hanketta on rahoittanut Etelä-Pohjanmaan liitto.

![epliitto_logo]

---

![seamk_logo]
