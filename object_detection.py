from transformers import pipeline
model = pipeline("object-detection")

print(model("cat.jpg"))

# [{'label': 'blanket',
#  'mask': mask_string,
#  'score': 0.917},
#...]
