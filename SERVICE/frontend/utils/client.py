import json
import os
import httpx
import asyncio


class RequestHandler:
    BASE_URL = os.environ.get("BACKEND_URL",
                              "http://localhost:8000") + "/api/v1"
    TIMEOUT = 5

    def __init__(self, logger):
        self.logger = logger

    async def fetch_one(
            self,
            method: str = 'POST',
            endpoint: str = '/',
            params=None,
            content=None,
            data=None,
            files=None,
            json=None
    ) -> dict[str, httpx.Response]:
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                response = await client.request(
                    method=method,
                    url=self.BASE_URL+endpoint,
                    params=params,
                    content=content,
                    data=data,
                    files=files,
                    json=json
                )
                response.raise_for_status()
                self.logger.info(f"Successfuly called '{response.url}'")
                return {"is_success": True, "response": response}
            except httpx.HTTPStatusError as e:
                self.logger.error(f"Error response {e.response.status_code} \
                                    while requesting {e.request.url}: \
                                    {e.response.text}")
                return {"is_success": False, "response": e.response}

    def get_models(self):
        endpoint = "/models/"
        return asyncio.run(
            self.fetch_one("GET", endpoint=endpoint))

    def load_model(self, model_id):
        endpoint = "/models/load"
        data = {"id": model_id}
        return asyncio.run(
            self.fetch_one("POST", endpoint=endpoint, json=data))

    def unload_model(self, model_id):
        endpoint = "/models/unload"
        data = {"id": model_id}
        return asyncio.run(
            self.fetch_one("POST", endpoint=endpoint, json=data))

    def predict(self, model_id, X):
        endpoint = f"/models/predict/{model_id}"
        data = {"X": X}
        return asyncio.run(
            self.fetch_one("POST", endpoint=endpoint, json=data))

    def train_model(
            self,
            data: bytes,
            model_id: str,
            model_type: str,
            model_params: dict,
            vectorizer_type: str,
            vectorizer_params: dict,
            spacy_lemma_tokenizer: bool = False
    ):
        endpoint = "/models/fit"
        files = {"fit_file": ("archive.zip", data, "application/zip")}
        data = {
            "id": model_id,
            "vectorizer_type": vectorizer_type,
            "vectorizer_params": json.dumps(vectorizer_params),
            "ml_model_type": model_type,
            "ml_model_params": json.dumps(model_params),
            "spacy_lemma_tokenizer": spacy_lemma_tokenizer
        }
        return asyncio.run(
            self.fetch_one("POST", endpoint=endpoint, files=files, data=data))

    def predict_scores(self, ids, zipped_csv):
        endpoint = "/models/predict_scores/"
        files = {
            "predict_file": ("archive.zip", zipped_csv, "application/zip")
        }
        data = {"ids": ','.join(ids)}
        return asyncio.run(
            self.fetch_one("POST", endpoint=endpoint, files=files, data=data))

    def remove_model(self, id):
        endpoint = f"/models/remove/{id}"
        return asyncio.run(
            self.fetch_one("DELETE", endpoint=endpoint))

    def remove_all_models(self):
        endpoint = "/models/remove_all"
        return asyncio.run(
            self.fetch_one("DELETE", endpoint=endpoint))

    def get_bg_tasks(self):
        endpoint = "/tasks/"
        return asyncio.run(
            self.fetch_one("GET", endpoint=endpoint))
