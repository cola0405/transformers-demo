# 认出来的就分好区域，然后打上各区域的标签

from transformers import pipeline

model = pipeline("image-segmentation")
res = model("cat.jpg")
print(res)
res[0]['mask'].show()