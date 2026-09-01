# Configurar la conexion a Google Sheets (una sola vez)

## 1. Crear un proyecto y activar la API
1. Ve a https://console.cloud.google.com/
2. Arriba a la izquierda, crea un proyecto nuevo (o usa uno existente). Cualquier nombre sirve, ej. "apuestas-mlb".
3. En el buscador superior escribe "Google Sheets API" > entra > click en **Habilitar**.

## 2. Crear la cuenta de servicio
1. Menu izquierdo: **IAM y administracion > Cuentas de servicio**.
2. Click **Crear cuenta de servicio**. Nombre: `mlb-sheets-bot` (o el que quieras). Click **Crear y continuar**, luego **Listo** (no necesitas asignar roles).
3. Ya creada, entra a la cuenta de servicio > pestana **Claves** > **Agregar clave > Crear clave nueva > JSON**.
4. Se descarga un archivo `.json`. Renombralo `credentials.json` y muevelo a esta carpeta (`apuestas_mlb/`).
   - **No lo subas a ningun repositorio publico** (ya esta en `.gitignore` por seguridad).

## 3. Compartir tu Google Sheet con la cuenta de servicio
1. Abre el archivo `credentials.json` y copia el valor de `"client_email"` (algo como
   `mlb-sheets-bot@tu-proyecto.iam.gserviceaccount.com`).
2. Abre tu Google Spreadsheet > boton **Compartir** > pega ese correo > dale permiso de **Editor** > Enviar.

## 4. Obtener el ID de tu spreadsheet
De la URL de tu hoja:
```
https://docs.google.com/spreadsheets/d/ESTE_ES_EL_ID/edit#gid=0
```
Copia la parte `ESTE_ES_EL_ID`.

## 5. Correr el programa
```
python mlb_first_inning_report.py "Reds" "Brewers" --sheet-id ESTE_ES_EL_ID --credentials credentials.json
```

O, para no escribir `--sheet-id` y `--credentials` cada vez, define variables de entorno:

PowerShell:
```
$env:MLB_SHEET_ID = "ESTE_ES_EL_ID"
$env:GOOGLE_SHEETS_CREDENTIALS = "C:\ruta\a\credentials.json"
python mlb_first_inning_report.py "Reds" "Brewers"
```

Cada corrida sobrescribe la pestana `Reporte MLB` (o la que definas con `--sheet-tab`) empezando en la celda `A1`, con una fila por metrica en formato `Etiqueta | Valor`. Si la pestana no existe, el programa la crea automaticamente.
