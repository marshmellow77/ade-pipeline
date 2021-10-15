import tarfile
import pathlib
import json
import botocore
from datasets.filesystems import S3FileSystem
from transformers import Trainer, AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if __name__ == "__main__":
    
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    model = AutoModelForSequenceClassification.from_pretrained(".")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    test_dataset = load_from_disk(f"/opt/ml/processing/test")
    
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
    report_dict = {
        "binary_classification_metrics": {
            "f1": {
                "value": eval_result['eval_f1'],
                "standard_deviation": 'NaN'
            },
            "accuracy": {
                "value": eval_result['eval_accuracy'],
                "standard_deviation": 'NaN'
            },
            "precision": {
                "value": eval_result['eval_precision'],
                "standard_deviation": 'NaN'
            },
            "recall": {
                "value": eval_result['eval_recall'],
                "standard_deviation": 'NaN'
            },
        },
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))