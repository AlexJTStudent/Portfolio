import json, base64, io
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

MODEL_PATH = "/opt/car_crash_model.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
CLASS_NAMES = {0: "without accident", 1: "with accident"}

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        img_b64 = body.get("image")
        if not img_b64:
            return {"statusCode": 400, "body": json.dumps({"error": "No image provided"})}

        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32)[np.newaxis, ...]
        arr = (arr / 127.5) - 1.0

        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        label = CLASS_NAMES[int(pred > 0.5)]

        return {"statusCode": 200, "body": json.dumps({"prediction": label})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
