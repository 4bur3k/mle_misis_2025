import onnxruntime as ort
import numpy as np
import cv2


class DepthEstimator:
    def __init__(self, model_path: str, input_size=(256, 256)):
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CHW
        img = np.expand_dims(img, axis=0)  # NCHW
        return img

    def infer(self, image: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        depth = output[0, 0]  # Remove batch & channel dims -> HxW
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        return depth
