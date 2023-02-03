import io
import argparse
import os
from PIL import Image

from flask import Flask, render_template, request, jsonify, redirect
import json
import pandas as pd
import torch

import mysql.connector

app = Flask(__name__)


def update_db(remote_addr, img_name, json_pred):
    rds_db = mysql.connector.connect(
    host = os.getenv('DB_HOSTNAME'),
    user = os.getenv('DB_USERNAME'),
    password = os.getenv('DB_PASSWORD'),
    database = "yolov5_predictions"
    )

    cur = rds_db.cursor()

    sql = "INSERT INTO requests (Address, ImageName, JsonPredictions) VALUES (%s, %s, %s)"
    val = (str(remote_addr), str(img_name), str(json_pred))
    cur.execute(sql, val)

    rds_db.commit()

@app.route('/save-labels', methods=['POST'])
def save_labels():
    request_data = json.loads(request.data)
    model_index = request_data['model_index']
    image_name = request_data['img_name']
    bounding_boxes = request_data['boundingBoxes']
    print("Image: " + image_name)
    if not bounding_boxes:
        print("The bounding boxes array is empty, saving original labels to S3")
        # Write './labels/predicted/' + image_name.rsplit('.', 1)[0] + '.txt' to S3
        # Write './labels/predicted/' + image_name.rsplit('.', 1)[0] + '.txt' to S3
    else:
        custom_bounding_boxes = pd.DataFrame(bounding_boxes)[['class_index', 'centerX', 'centerY', 'boxWidth', 'boxHeight']]
        custom_labels_path = './labels/custom/' + model_index + "/" + image_name.rsplit('.', 1)[0] + '.txt'
        custom_bounding_boxes.to_csv(path_or_buf=custom_labels_path, sep=' ', header=False, index=False)
        # Write custom_labels_path to S3
    #print(bounding_boxes)
    #print(bounding_boxes[0])

    return jsonify({"message": "Bounding box saved successfully"}), 200

# Load image from user
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        model_index = "0"
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Save image file
        original_image_loc = "static/original/" + model_index + "/" + image_name
        img.save(original_image_loc)

        results = model(img, size=640)
        pred_bounding_boxes = results.pandas().xywhn[0][['class', 'xcenter', 'ycenter', 'width', 'height']]
        # Save predictions to txt
        pred_labels_path = './labels/predicted/' + model_index + "/" + image_name.rsplit('.', 1)[0] + '.txt'
        pred_bounding_boxes.to_csv(path_or_buf=pred_labels_path, sep=' ', header=False, index=False)
        json_pred = pred_bounding_boxes.to_json(orient="records")

        results.render()
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            predicted_image_loc = "static/predictions/" + model_index + "/predict_" + image_name
            img_base64.save(predicted_image_loc, format="JPEG")
            if 'DB_HOSTNAME' in os.environ and 'DB_USERNAME' in os.environ and 'DB_PASSWORD' in os.environ:
                update_db(request.remote_addr, image_name, json_pred)
        #return redirect(predicted_image_name)
        return render_template("index.html", pred_image_loc=predicted_image_loc, orig_image_loc=original_image_loc)

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web app for YOLOv5")
    parser.add_argument("--port", default=32332, type=int, help="Service port")
    parser.add_argument("--model", default="YCVR_big", type=str, help="Model to use: YCVR_big, YCVR_small, YCR_big, YCR_small, YCV.")
    args = parser.parse_args()
    model_path = "./models/" + args.model + ".pt"
    model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local', autoshape=True)
    model.eval()
    model.cpu()
    app.run(host="0.0.0.0", port=args.port)
