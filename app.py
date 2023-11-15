import boto3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY_ID = os.environ.get('AWS_SECRET_KEY_ID')
AWS_REGION = os.environ.get('AWS_REGION')
S3_BUCKET_NAME = "ask-trained-models"

BASE_MODEL = "codellama/CodeLlama-34b-Python-hf"
S3_FOLDER_PREFIX = "FusedModel"
TRAINING_ID = "488a0fda-3992-4dbe-91e6-d0383e8ef5eb"



CHECKPOINT = "1700"
S3_FOLDER_PATH = f"{S3_FOLDER_PREFIX}/{TRAINING_ID}"



class InferlessPythonModel:
    def initialize(self):
        s3 = boto3.client(
            's3', 
            aws_access_key_id=AWS_ACCESS_KEY_ID, 
            aws_secret_access_key=AWS_SECRET_KEY_ID, 
            region_name=AWS_REGION
        )

        objects = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_FOLDER_PATH)
        for obj in objects["Contents"]:
            key = obj["Key"]
            file_name = os.path.join(TRAINING_ID, key.replace(S3_FOLDER_PATH + "/", ""))
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            s3.download_file(S3_BUCKET_NAME, key, file_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(TRAINING_ID)
        self.model = AutoModelForCausalLM.from_pretrained(TRAINING_ID, torch_dtype=torch.float16, device_map="cuda:0")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=600)
        result = self.tokenizer.decode(output[0])
        return {"generated_result": "Hello there"}

    def finalize(self,args):
        self.tokenizer = None
        self.model = None
