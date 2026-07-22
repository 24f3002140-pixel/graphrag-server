# GraphRAG Pipeline

## Endpoints

- `POST /extract-graph`
- `POST /graph-query`
- `POST /community-summary`
- `GET /health`

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: `http://127.0.0.1:8000/docs`

## Deploy on Render

1. Upload these files to a GitHub repository.
2. In Render, create a **New Web Service** from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. After deployment, submit only the root URL, for example:
   `https://graphrag-pipeline.onrender.com`

Do not include `/extract-graph` in the submitted base URL.

## Colab + ngrok alternative

```python
!pip install -r requirements.txt pyngrok
```

```python
from pyngrok import ngrok
import subprocess, time

process = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
)
time.sleep(3)
public_url = ngrok.connect(8000)
print(public_url)
```

Submit the printed HTTPS root URL.
