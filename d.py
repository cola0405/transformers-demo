from transformers import pipeline

model = pipeline("bert-base-uncased")

print(model('hello man'))