from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/long-t5-tglobal-base"
cache_dir = "./models"

print("Downloading model... This may take a few minutes.")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("✅ Model downloaded and cached successfully!")
