from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection


path = r'C:\Users\Cola\Desktop\braille.jpg'
im = Image.open(path)
# 提取出n个object
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
encoding = feature_extractor(im, return_tensors="pt")
encoding.keys()


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
# 他们自定义的output
outputs = model(**encoding)


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        # 最大可能的index
        cl = p.argmax()

        # id2label
        # p[cl]是对于的可能性数值
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'

        # 标记相关信息
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()




# keep only predictions of queries with 0.9+ confidence (excluding no-object class)
# 第二个-1表示不取最后一个
# probas是二维的
# 表示图像中含有n个object，对于每个object都做91种标签的预测
probas = outputs.logits.softmax(-1)[0, :, :-1]


# values是一维的！n个object的预测的可能性max的序列
# keep 将用于保留max>0.9的那些行（行内含91个标签的可能性）
# keep 是1维的
keep = probas.max(-1).values > 0.9

# rescale bounding boxes
target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)


# keep是一维的，boxes是二维的——子项是四个值，分别是左下和右上的坐标
# 那么会取到为True的那一行
# boxes中的一项：[ 4.9941e+02,  1.7105e+02,  5.1202e+02,  5.0936e+02]
# 取到True的那一行的点的坐标
bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]


# probas[keep] 是max>0.9的那些行（行内含91个标签的可能性）
# 可能有多行是因为可能检测到多个object
plot_results(im, probas[keep], bboxes_scaled)
