  # Dokumentacja Frontendu

  Aplikacja działa w przeglądarce i jest zbudowana w React, TypeScript oraz Vite.

  ## Wymagania

  - Node.js i npm

  ## Uruchomienie lokalne

  W terminalu przejdź do katalogu `FE`, zainstaluj zależności i uruchom serwer developerski:

  ```bash
  cd FE
  npm ci
  npm run dev
  ```

  Vite wyświetli adres aplikacji w terminalu, zwykle `http://localhost:5173`. Otwórz ten adres w przeglądarce. Serwer zatrzymasz skrótem `Ctrl+C`.

  Jeśli terminal jest już otwarty w katalogu `FE`, pomiń polecenie `cd FE`.

  ## Pozostałe polecenia

  Uruchom z katalogu `FE`:

  ```bash
  npm run build    # sprawdza typy i tworzy wersję produkcyjną w katalogu dist
  npm run lint     # sprawdza kod za pomocą ESLint
  npm run preview  # lokalnie serwuje wcześniej zbudowaną wersję
  ```

  Przed `npm run preview` wykonaj `npm run build`.

