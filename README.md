# Prisme — analyse de portefeuille

Une application web locale en Python pour analyser un portefeuille client et le comparer à un benchmark personnalisable.

## Ce que la V1 permet

- Saisir des actions, FNB, fonds mutuels et produits obligataires par symbole Yahoo Finance.
- Définir les pondérations et la devise de chaque position.
- Convertir automatiquement les actifs CAD et USD dans la devise d'analyse.
- Choisir une période, un taux sans risque et une fréquence de rééquilibrage.
- Construire un benchmark avec huit blocs : S&P 500, Nasdaq 100, Dow Jones, Canada/TSX,
  MSCI international, obligations US, obligations canadiennes et obligations internationales.
- Afficher rendement total et annualisé, volatilité, Sharpe, Sortino, Calmar, baisse maximale,
  VaR/CVaR, alpha, bêta, corrélation, erreur de suivi et ratios de capture.
- Visualiser la croissance d'un placement et les drawdowns du portefeuille et du benchmark.
- Consulter les rendements par année et exporter les séries quotidiennes en CSV.

## Installation dans VS Code (Windows)

Ouvrez le dossier du projet dans VS Code, puis un terminal PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dans VS Code, sélectionnez ensuite l'interpréteur Python situé dans `.venv` si l'éditeur ne le fait pas automatiquement.

## Lancement

```powershell
python -m streamlit run app.py
```

Streamlit ouvre normalement l'application dans le navigateur. Sinon, utilisez l'adresse locale affichée dans le terminal,
habituellement `http://localhost:8501`.

Sur Windows, l'application fusionne automatiquement les certificats de confiance du système avec ceux de Python. Si tous
les symboles échouent encore avec une erreur SSL, Yahoo Finance est probablement bloqué par le réseau; faites approuver
le domaine par votre organisation plutôt que de désactiver la validation SSL.

## Utilisation

1. Remplacez le portefeuille d'exemple par les positions du client.
2. Vérifiez que les pondérations totalisent 100 %.
3. Répartissez 100 % entre les blocs du benchmark.
4. Choisissez la devise et la période dans le panneau de gauche.
5. Cliquez sur **Analyser le portefeuille**.

### Symboles

- États-Unis : `AAPL`, `SPY`, `AGG`.
- Canada : `RY.TO`, `XIC.TO`, `XBB.TO`.
- Fonds mutuels : le symbole doit exister dans Yahoo Finance. La couverture des fonds canadiens est inégale.
- Obligations individuelles : uniquement lorsqu'un historique Yahoo Finance existe. Les FNB obligataires sont plus fiables.

Les prix sont ajustés pour tenir compte des distributions et fractionnements. La source Yahoo Finance accessible par
`yfinance` est destinée à la recherche et à un usage personnel; vérifiez ses conditions avant tout usage commercial.

## Hypothèses importantes

- Les pondérations saisies sont des pondérations cibles, rétablies selon la fréquence choisie.
- Toutes les séries commencent à la première date commune disponible. Un titre récent peut donc raccourcir l'analyse.
- La conversion USD/CAD utilise le symbole `CAD=X`.
- L'analyse n'inclut pas les flux réels du client, les frais, l'impôt ni les coûts de transaction.
- La VaR et la CVaR sont historiques; elles ne représentent pas une perte maximale garantie.

## Tests

Les calculs financiers peuvent être vérifiés sans connexion Internet :

```powershell
python -m unittest discover -s tests -v
```

## Suite logique

La prochaine phase pourra ajouter l'import Excel, la comparaison avec vos portefeuilles modèles, les rendements pondérés
par les flux réels, un rapport PDF client et une source professionnelle pour élargir la couverture des fonds mutuels.
avoir une option de benchmark custom avec un portefeuille ou l'on indique les stocks, etf, fonds mutuels comme la section du depart. les fonds mutuels canadien ne sont pas sur yahoofinance, voir si morningstar pourrait etre fetche ou crawl
ajouter la possibilite d'enregistrer un portefeuille modele et d'en charger un.
