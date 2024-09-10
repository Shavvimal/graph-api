# Graph API 

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker 

```bash
docker build -t graphrag-api .
```

To run, specifying with our `.env` file:

```bash
docker run graphrag-api --env-file <env_file>
```

For me:

```bash
docker run --env-file docker.env -p 8000:8000 graphrag-api -name graphrag-api
```
