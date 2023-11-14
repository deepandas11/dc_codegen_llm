import boto3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY_ID = os.environ.get('AWS_SECRET_KEY_ID')
AWS_REGION = os.environ.get('AWS_REGION')
S3_FOLDER_PREFIX = "Train-Versioning"
S3_BUCKET_NAME = "ask-trained-models"


BASE_MODEL = "codellama/CodeLlama-34b-Python-hf"

TRAINING_ID = "488a0fda-3992-4dbe-91e6-d0383e8ef5eb"
CHECKPOINT = "1700"
FOLDER_NAME = f"{S3_FOLDER_PREFIX}/{TRAINING_ID}/Checkpoints/checkpoint-{CHECKPOINT}"

LORA_PATH= f"lora_model/{TRAINING_ID}/"

HF_KEY = os.environ.get('HF_KEY')



class InferlessPythonModel:
    def initialize(self):
        s3 = boto3.client(
            's3', 
            aws_access_key_id=AWS_ACCESS_KEY_ID, 
            aws_secret_access_key=AWS_SECRET_KEY_ID, 
            region_name=AWS_REGION
        )

        objects = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=FOLDER_NAME)
        for obj in objects["Contents"]:
            key = obj["Key"]
            file_name = os.path.join(LORA_PATH, key.replace(FOLDER_NAME + "/", ""))
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            s3.download_file(S3_BUCKET_NAME, key, file_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cuda:0")
        self.model = PeftModel(self.model, LORA_PATH)
        self.model = self.model.merge_and_unload()



    def infer(self, inputs):
        prompt = inputs["prompt"]
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=600)
        result = self.tokenizer.decode(output[0])
        return {"generated_result": result}

    def finalize(self,args):
        self.tokenizer = None
        self.model = None