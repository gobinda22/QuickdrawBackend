
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

classes = ['Bird', 'Flower', 'Hand', 'House', 'Pencil', 'Spectacles', 'Spoon', 'Sun', 'Tree', 'Umbrella']

ort_session = ort.InferenceSession('C:/Users/Gobinda/Desktop/QDRML/savedModel/model.onnx')  # load the saved onnx model

# pre processing
def process(path):
    # pre process same as training
    image = Image.open(path)
    arr = np.asarray(image)
    image = Image.fromarray(arr[:, :, 3])  # read alpha channel
    image = image.resize((64, 64))
    image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]

    return image[None]


# tes the model
def test(path):
    image = process(path)
    output = ort_session.run(None, {'data': image})[0].argmax()

    print(classes[output], output)

    return classes[output]

# if __name__=='__main__':
#     test(path)

