from serializers.trainer import MLModelType

DEFAULT_MODELS_INFO = {
    "default_logistic_regression": {
        "type": MLModelType.logistic_regression,
        "filename": "model_lr_e.cloudpickle",
        "is_dl_model": False
    },
    "default_linear_svc": {
        "type": MLModelType.linear_svc,
        "filename": "model_svc_e.cloudpickle",
        "is_dl_model": False
    },
    "default_multinomial_naive_bayes": {
        "type": MLModelType.multinomial_nb,
        "filename": "model_mnb_e.cloudpickle",
        "is_dl_model": False
    },
    "default_distilbert": {
        "type": MLModelType.distilbert,
        "filename": "distilbert",
        "is_dl_model": True
    }
}

loaded_models = {}


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class DLModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None

    def load_model(self, model_path):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    async def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1)
        
        return {
            "prediction": prediction.item(),
            "confidence": confidence.item()
        }

dl_model_manager = DLModelManager()
