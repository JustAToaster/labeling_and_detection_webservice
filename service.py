import io
import argparse
import os
from PIL import Image

from flask import Flask, render_template, request, jsonify, redirect
import json
import pandas as pd
import torch
import zipfile

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
    curr_model_pt = request_data['curr_model_pt']
    image_name = request_data['img_name']
    bounding_boxes = request_data['boundingBoxes']
    print("Image: " + image_name)
    if not bounding_boxes:
        print("The bounding boxes array is empty, saving predicted labels to S3")
        pred_labels_path = './labels/predicted/' + curr_model_pt + "/" + image_name.rsplit('.', 1)[0] + '.txt'
        # Write pred_labels_path to S3
    else:
        if not os.path.exists('./labels/custom/' + curr_model_pt):
            os.makedirs('./labels/custom/' + curr_model_pt)
        custom_labels_path = './labels/custom/' + curr_model_pt + "/" + image_name.rsplit('.', 1)[0] + '.txt'
        if bounding_boxes == "No bounding boxes":
            with open(custom_labels_path, 'w') as labels_file:
                labels_file.write("")
        else:
            custom_bounding_boxes = pd.DataFrame(bounding_boxes)[['class_index', 'centerX', 'centerY', 'boxWidth', 'boxHeight']]
            custom_bounding_boxes.to_csv(path_or_buf=custom_labels_path, sep=' ', header=False, index=False)
        # Write custom_labels_path to S3
    #print(bounding_boxes)
    #print(bounding_boxes[0])

    return jsonify({"message": "Bounding box saved successfully"}), 200

# Load image from user
@app.route("/", methods=["GET", "POST"])
def predict():
    models_list = os.listdir('./models')
    curr_model_pt = models_list[0].rsplit('.', 1)[0]
    model = torch.hub.load('./yolov5', 'custom', path="./models/" + args.model + ".pt", source='local', autoshape=True)
    model.eval()
    model.cpu()
    num_training_images = 0
    # TODO: get training set size from S3

    if request.method == "POST":
        curr_model_pt = request.form['model_selection'].rsplit('.', 1)[0]
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file:
            return

        # Load requested model
        model = torch.hub.load('./yolov5', 'custom', path="./models/" + curr_model_pt + ".pt", source='local', autoshape=True)
        print("Loaded model with classes: ")
        print(model.names)
        model.eval()
        model.cpu()

        num_training_images = 0
        # TODO: get training set size from S3

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Save image to temp static folder
        static_model_path = "static/original/" + curr_model_pt + "/"
        if not os.path.exists(static_model_path):
            os.makedirs(static_model_path)
        original_image_loc = static_model_path + image_name
        img.save(original_image_loc)

        results = model(img, size=640)
        pred_bounding_boxes = results.pandas().xywhn[0][['class', 'xcenter', 'ycenter', 'width', 'height']]
        # Save predictions to txt
        if not os.path.exists('./labels/predicted/' + curr_model_pt):
            os.makedirs('./labels/predicted/' + curr_model_pt)
        pred_labels_path = './labels/predicted/' + curr_model_pt + "/" + image_name.rsplit('.', 1)[0] + '.txt'
        pred_bounding_boxes.to_csv(path_or_buf=pred_labels_path, sep=' ', header=False, index=False)
        json_pred = pred_bounding_boxes.to_json(orient="records")

        results.render()
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            predicted_image_loc = "static/predictions/" + curr_model_pt + "/predict_" + image_name
            img_base64.save(predicted_image_loc, format="JPEG")
            if 'DB_HOSTNAME' in os.environ and 'DB_USERNAME' in os.environ and 'DB_PASSWORD' in os.environ:
                update_db(request.remote_addr, image_name, json_pred)
        #return redirect(predicted_image_name)
        return render_template("index.html", pred_image_loc=predicted_image_loc, orig_image_loc=original_image_loc, class_names=model.names, num_classes=len(model.names), models=models_list, num_models=len(models_list), curr_model_pt=curr_model_pt, num_training_images=num_training_images)

    return render_template("index.html", class_names=model.names, num_classes=len(model.names), models=models_list, num_models=len(models_list), curr_model_pt=curr_model_pt, num_training_images=num_training_images)

# Request model form
@app.route("/request_model", methods=["GET", "POST"])
def request_model():

    if request.method == "POST":
        model_name = request.form['model_name']
        class_list = request.form['class_list']
        pending_model_path = './pending_models/' + model_name
        if not os.path.exists(pending_model_path + '/labels/train'):
            os.makedirs(pending_model_path + '/labels/train')
        if not os.path.exists(pending_model_path + '/images/train'):
            os.makedirs(pending_model_path + '/images/train')
        # Write class list
        with open(pending_model_path + '/classes.txt', 'w') as class_list_file:
            class_list_file.write(class_list)
        if "file" in request.files and request.files["file"]:
            file = request.files["file"]
            file_like_object = file.stream._file  
            zipfile_ob = zipfile.ZipFile(file_like_object)
            zipfile_ob.extractall(path=pending_model_path)
            # Upload files to S3 ...
        return render_template("request_model.html")

    return render_template("request_model.html")

# Send data for pending models
@app.route("/pending_models", methods=["GET", "POST"])
def pending_models():
    pending_models_list = os.listdir('./pending_models')
    if not pending_models_list:
        no_pending_models = True
        return render_template("pending_models.html", no_pending_models=no_pending_models)
    curr_pending_model = pending_models_list[0].rsplit('.', 1)[0]
    with open('./pending_models/' + curr_pending_model + '/classes.txt', 'r') as class_list_file:
            class_list = list(filter(None, class_list_file.read().split('\n')))
    num_training_images = 0
    # TODO: get training set size from S3
    min_training_set_size = 100

    if request.method == "POST":
        curr_pending_model = request.form['model_selection'].rsplit('.', 1)[0]
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file:
            return
        pending_model_path = './pending_models/' + curr_pending_model
        with open(pending_model_path + '/classes.txt', 'r') as class_list_file:
            class_list = list(filter(None, class_list_file.read().split('\n')))
        num_training_images = len(os.listdir('./pending_models/' + curr_pending_model + '/labels/train'))
        # TODO: get training set size from S3

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Save image to temp static folder
        static_model_path = "static/original/" + curr_pending_model + "/"
        if not os.path.exists(static_model_path):
            os.makedirs(static_model_path)
        original_image_loc = static_model_path + image_name
        img.save(original_image_loc)

        return render_template("pending_models.html", orig_image_loc=original_image_loc, class_names=class_list, num_classes=len(class_list), pending_models=pending_models_list, num_pending_models=len(pending_models_list), curr_pending_model=curr_pending_model, num_training_images=num_training_images, min_training_set_size=min_training_set_size)

    return render_template("pending_models.html", class_names=class_list, num_classes=len(class_list), pending_models=pending_models_list, num_pending_models=len(pending_models_list), curr_pending_model=curr_pending_model, num_training_images=num_training_images, min_training_set_size=min_training_set_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web app for YOLOv5")
    parser.add_argument("--port", default=32332, type=int, help="Service port")
    parser.add_argument("--model", default="base_yolov5s", type=str, help="Default model to use")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
