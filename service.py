import io
import argparse
import os
from PIL import Image

from random import random

from flask import Flask, render_template, request, jsonify, redirect
import json
import pandas as pd
import torch
import zipfile
import boto3

import psycopg2

from custom_metrics import customization_score

app = Flask(__name__)

s3_interaction = 0 if os.getenv('S3_INTERACTION') == '0' else 1

if s3_interaction:
    s3_resource = boto3.resource('s3')
    s3_client = boto3.client('s3')

def download_single_file_if_updated(bucket_name, filename):
    obj = s3_resource.Object(bucket_name, filename)
    remote_last_modified = int(obj.last_modified.strftime('%s'))
    if os.path.exists(obj.key) and remote_last_modified == int(os.path.getmtime(obj.key)):
            print("File " + obj.key + " is up to date")
    else:
        print("Downloading " + filename + " from the S3 bucket")
        obj.download_file(filename)
        os.utime(obj.key, (remote_last_modified, remote_last_modified))

def download_files_if_updated(bucket_name, s3_folder):
    bucket = s3_resource.Bucket(bucket_name)
    if not os.path.exists(s3_folder):
        os.makedirs(s3_folder)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        remote_last_modified = int(obj.last_modified.strftime('%s'))
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        if obj.key[-1] == '/':
            continue
        if os.path.exists(obj.key) and remote_last_modified == int(os.path.getmtime(obj.key)):
            print("File " + obj.key + " is up to date")
        else:
            print("Downloading " + obj.key + " from the S3 bucket")
            bucket.download_file(obj.key, obj.key)
            os.utime(obj.key, (remote_last_modified, remote_last_modified))

def get_pending_models(bucket_name, s3_folder, txt_file):
    bucket = s3_resource.Bucket(bucket_name)
    result = s3_client.list_objects(Bucket=bucket.name, Prefix=s3_folder, Delimiter='/')
    pending_models_list = [o.get('Prefix').split('/')[1] for o in result.get('CommonPrefixes')]
    for pending_model in pending_models_list:
        if not os.path.exists(s3_folder + pending_model):
            os.makedirs(s3_folder + pending_model)
        class_file_name = s3_folder + pending_model + txt_file
        remote_last_modified = int(s3_resource.ObjectSummary(bucket_name=bucket.name, key=class_file_name).last_modified.strftime('%s'))
        if os.path.exists(class_file_name) and remote_last_modified == int(os.path.getmtime(class_file_name)):
            print("File " + class_file_name + " is up to date")
        else:
            print("Downloading " + class_file_name + " from the S3 bucket")
            bucket.download_file(class_file_name, class_file_name)
            os.utime(class_file_name, (remote_last_modified, remote_last_modified))

def upload_pending_model(bucket_name, pending_model_path):
    print("Uploading class list for " + pending_model_path + " to the S3 bucket")
    s3_client.upload_file(pending_model_path + 'classes.txt', bucket_name, pending_model_path + 'classes.txt')
    for labels_filename in os.listdir(pending_model_path + 'labels/'):
        labels_filepath = os.path.join(pending_model_path + 'labels/', labels_filename)
        image_filepath = labels_filepath.rsplit('.', 1)[0] + '.jpg'
        image_filename = labels_filename.rsplit('.', 1)[0] + '.jpg'
        extracted_set = "/"
        # Randomize training and validation split
        if random() <= 0.8:
            extracted_set = '/train/'
        else:
            extracted_set = '/valid/'
        print("Uploading image and labels for " + image_filename + " to the S3 bucket")
        s3_client.upload_file(labels_filepath, bucket_name, pending_model_path + extracted_set + 'labels/' + labels_filename)
        s3_client.upload_file(image_filepath, bucket_name, pending_model_path + extracted_set + 'images/' + image_filename)

# Use paginator in case training set size > 1000 (S3 API limit)
def count_dataset_size(bucket_name, model_path):
    paginator = s3_client.get_paginator('list_objects_v2')
    training_count = 0
    validation_count = 0
    #print(model_path)
    for result in paginator.paginate(Bucket=bucket_name, Prefix=model_path + '/labels/train/', Delimiter='/'):
        key_count = result.get('KeyCount')
        if key_count is not None:
            training_count += key_count

    for result in paginator.paginate(Bucket=bucket_name, Prefix=model_path + '/labels/valid/', Delimiter='/'):
        key_count = result.get('KeyCount')
        if key_count is not None:
            validation_count += key_count

    return training_count, validation_count

def update_db(remote_addr, img_name, model_name, cust_score):
    conn = psycopg2.connect(
    host=os.getenv('DB_HOSTNAME'),
    database="yolov5_predictions",
    user=os.getenv('DB_USERNAME'),
    password=os.getenv('DB_PASSWORD'))

    cur = conn.cursor()
    sql = "INSERT INTO requests (UserAddress, ModelName, ImageName, CustomizationScore) VALUES (%s, %s, %s)"
    val = (str(remote_addr), str(model_name), str(img_name), str(cust_score))
    cur.execute(sql, val)

    conn.commit()
    cur.close()
    conn.close()

def check_user(user_addr):
    if s3_interaction:
        download_single_file_if_updated(args.models_bucket, 'report_list.txt')
    with open('report_list.txt', 'r') as report_list_file:
        reported_users_list = [line.split('\t')[0] for line in list(filter(None, report_list_file.read().split('\n')))]
    if user_addr in reported_users_list:
        print("The reported user with address " + user_addr + " is trying to access the website")
        return True
    return False

@app.route('/save_labels', methods=['POST'])
def save_labels():
    if check_user(request.remote_addr):
        return render_template("reported.html")
    request_data = json.loads(request.data)
    curr_model = request_data['curr_model']
    image_name = request_data['img_name']
    bounding_boxes = request_data['boundingBoxes']
    is_pending_model = request_data['is_pending_model']
    model_folder_prefix = ""
    if is_pending_model:
        model_folder_prefix = "pending_"
    labels_txt = image_name.rsplit('.', 1)[0] + '.txt'
    static_model_path = "static/original/" + curr_model + "/"
    predicted_labels_path = 'labels/predicted/' + curr_model + "/" + labels_txt
    custom_labels_path = 'labels/custom/' + curr_model + "/" + labels_txt
    image_filepath = static_model_path + image_name
    extracted_set = "/"
    # Randomize training and validation split
    if random() <= 0.8:
        extracted_set = '/train/'
    else:
        extracted_set = '/valid/'
    if not bounding_boxes:
        # Remove confidence from predicted labels
        predicted_bounding_boxes = pd.read_csv(filepath_or_buffer=predicted_labels_path, sep=' ', names=['class_index', 'centerX', 'centerY', 'boxWidth', 'boxHeight', 'confidence'], index_col=False).drop('confidence', axis=1).to_csv(path_or_buf=predicted_labels_path, sep=' ', header=False, index=False)
        if s3_interaction:
            print("The bounding boxes array is empty, saving predicted labels for " + image_name + " to S3")
            s3_client.upload_file(predicted_labels_path, args.training_data_bucket, model_folder_prefix + 'models/' + curr_model + '/labels' + extracted_set + labels_txt)
    else:
        if not os.path.exists('labels/custom/' + curr_model):
            os.makedirs('labels/custom/' + curr_model)
        # If the user did not create any boxes but customized, create an empty file
        if bounding_boxes == "No bounding boxes":
            with open(custom_labels_path, 'w') as labels_file:
                labels_file.write("")
        else:
            custom_bounding_boxes = pd.DataFrame(bounding_boxes)[['class_index', 'centerX', 'centerY', 'boxWidth', 'boxHeight']]
            custom_bounding_boxes.to_csv(path_or_buf=custom_labels_path, sep=' ', header=False, index=False)
        # If user customized labels, compute a metric to evaluate how likely he is sending malicious custom labels
        if not is_pending_model:
            predicted_bounding_boxes = pd.read_csv(filepath_or_buffer=predicted_labels_path, sep=' ', names=['class_index', 'centerX', 'centerY', 'boxWidth', 'boxHeight', 'confidence'], index_col=False).to_dict(orient='records')
            val_AP_classes = []
            if os.path.isfile('models/' + curr_model + '/val_AP.txt'):
                with open('models/' + curr_model + '/val_AP.txt', 'r') as val_AP_file:
                    val_AP_classes = [float(x) for x in list(filter(None, val_AP_file.read().split('\n')))]
            cust_score = customization_score(predicted_bounding_boxes, bounding_boxes, val_AP_classes)
            print("Customization score: " + str(cust_score))
            if 'DB_HOSTNAME' in os.environ and 'DB_USERNAME' in os.environ and 'DB_PASSWORD' in os.environ:
                update_db(request.remote_addr, curr_model, image_name, cust_score)
        if s3_interaction:
            print("Saving customized labels for " + image_name + " to S3")
            s3_client.upload_file(custom_labels_path, args.training_data_bucket, model_folder_prefix + 'models/' + curr_model + '/labels' + extracted_set + labels_txt)
    
    if s3_interaction:
        print("Uploading image " + image_name + " to the S3 bucket")
        s3_client.upload_file(image_filepath, args.training_data_bucket, model_folder_prefix + 'models/' + curr_model + '/images' + extracted_set + image_name)
    return jsonify({"message": "Bounding boxes saved successfully"}), 200

# Load image from user
@app.route("/", methods=["GET", "POST"])
def predict():
    if s3_interaction:
        download_files_if_updated(args.models_bucket, 'models')
    if check_user(request.remote_addr):
        return render_template("reported.html")
    models_list = os.listdir('models')
    curr_model = models_list[0]
    model = torch.hub.load('yolov5', 'custom', path="models/" + args.model + "/" + args.model + ".pt", source='local', autoshape=True)
    model.eval()
    model.cpu()
    num_training_images, num_validation_images = 0, 0
    if s3_interaction:
        num_training_images, num_validation_images = count_dataset_size(args.training_data_bucket, "models/" + curr_model)

    if request.method == "POST":
        curr_model = request.form['model_selection']
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file:
            return redirect(request.url)

        # Load requested model
        model = torch.hub.load('yolov5', 'custom', path="models/" + curr_model + "/" + curr_model + ".pt", source='local', autoshape=True)
        print("Loaded model with classes: ")
        print(model.names)
        model.eval()
        model.cpu()

        num_training_images, num_validation_images = 0, 0
        if s3_interaction:
            num_training_images, num_validation_images = count_dataset_size(args.training_data_bucket, "models/" + curr_model)

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Save image to temp static folder
        original_images_folder = "static/original/" + curr_model + "/"
        if not os.path.exists(original_images_folder):
            os.makedirs(original_images_folder)
        original_image_loc = original_images_folder + image_name
        img.save(original_image_loc)

        results = model(img, size=640)
        pred_bounding_boxes = results.pandas().xywhn[0][['class', 'xcenter', 'ycenter', 'width', 'height', 'confidence']]
        # Save predictions to txt
        if not os.path.exists('labels/predicted/' + curr_model):
            os.makedirs('labels/predicted/' + curr_model)
        pred_labels_path = 'labels/predicted/' + curr_model + "/" + image_name.rsplit('.', 1)[0] + '.txt'
        pred_bounding_boxes.to_csv(path_or_buf=pred_labels_path, sep=' ', header=False, index=False)
        # json_pred = pred_bounding_boxes.to_json(orient="records")
        results.render()

        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            predicted_images_folder = "static/predictions/" + curr_model + "/"
            if not os.path.exists(predicted_images_folder):
                os.makedirs(predicted_images_folder)
            predicted_image_loc = predicted_images_folder + "/predict_" + image_name
            img_base64.save(predicted_image_loc, format="JPEG")
        #return redirect(predicted_image_name)
        return render_template("index.html", pred_image_loc=predicted_image_loc, orig_image_loc=original_image_loc, class_names=model.names, num_classes=len(model.names), models=models_list, num_models=len(models_list), curr_model=curr_model, num_training_images=num_training_images, num_validation_images=num_validation_images)

    return render_template("index.html", class_names=model.names, num_classes=len(model.names), models=models_list, num_models=len(models_list), curr_model=curr_model, num_training_images=num_training_images, num_validation_images=num_validation_images)

# Request model form
@app.route("/request_model", methods=["GET", "POST"])
def request_model():
    if check_user(request.remote_addr):
        return render_template("reported.html")

    if request.method == "POST":
        model_name = request.form['model_name']
        class_list = request.form['class_list']
        pending_model_path = 'pending_models/' + model_name + '/'
        # Write class list
        with open(pending_model_path + 'classes.txt', 'w') as class_list_file:
            class_list_file.write(class_list)
        if "file" in request.files and request.files["file"]:
            file = request.files["file"]
            file_like_object = file.stream._file  
            zipfile_ob = zipfile.ZipFile(file_like_object)
            zipfile_ob.extractall(path=pending_model_path)
        if s3_interaction:
            upload_pending_model(args.training_data_bucket, pending_model_path)
        return render_template("request_model.html")

    return render_template("request_model.html")

# Send data for pending models
@app.route("/pending_models", methods=["GET", "POST"])
def pending_models():
    if check_user(request.remote_addr):
        return render_template("reported.html")
    
    if s3_interaction:
        get_pending_models(args.training_data_bucket, 'pending_models/', '/classes.txt')
    if not os.path.exists('pending_models'):
        os.makedirs('pending_models')
    pending_models_list = os.listdir('pending_models')
    if not pending_models_list:
        no_pending_models = True
        return render_template("pending_models.html", no_pending_models=no_pending_models)
    curr_pending_model = pending_models_list[0].rsplit('.', 1)[0]
    with open('pending_models/' + curr_pending_model + '/classes.txt', 'r') as class_list_file:
            class_list = list(filter(None, class_list_file.read().split('\n')))
    num_training_images, num_validation_images = 0, 0
    if s3_interaction:
        num_training_images, num_validation_images = count_dataset_size(args.training_data_bucket, "models/" + curr_pending_model)
    min_training_set_size = 80
    min_validation_set_size = 20

    if request.method == "POST":
        curr_pending_model = request.form['model_selection'].rsplit('.', 1)[0]
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file:
            return
        pending_model_path = 'pending_models/' + curr_pending_model + '/'
        with open(pending_model_path + 'classes.txt', 'r') as class_list_file:
            class_list = list(filter(None, class_list_file.read().split('\n')))
        
        if s3_interaction:
            num_training_images, num_validation_images = count_dataset_size(args.training_data_bucket, "models/" + curr_pending_model)

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Save image to temp static folder
        static_model_path = "static/original/" + curr_pending_model + "/"
        if not os.path.exists(static_model_path):
            os.makedirs(static_model_path)
        original_image_loc = static_model_path + image_name
        img.save(original_image_loc)

        return render_template("pending_models.html", orig_image_loc=original_image_loc, class_names=class_list, num_classes=len(class_list), pending_models=pending_models_list, num_pending_models=len(pending_models_list), curr_pending_model=curr_pending_model, num_training_images=num_training_images, num_validation_images=num_validation_images, min_training_set_size=min_training_set_size, min_validation_set_size=min_validation_set_size)

    return render_template("pending_models.html", class_names=class_list, num_classes=len(class_list), pending_models=pending_models_list, num_pending_models=len(pending_models_list), curr_pending_model=curr_pending_model, num_training_images=num_training_images, num_validation_images=num_validation_images, min_training_set_size=min_training_set_size, min_validation_set_size=min_validation_set_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web app for YOLOv5")
    parser.add_argument("--port", default=32332, type=int, help="Service port")
    parser.add_argument("--model", default="base_yolov5s", type=str, help="Default model to use")
    parser.add_argument("--training_data_bucket", default="justatoaster-yolov5-training-data", type=str, help="Training data S3 bucket name")
    parser.add_argument("--models_bucket", default="justatoaster-yolov5-models", type=str, help="YOLOv5 models S3 bucket name")
    args = parser.parse_args()
    #models_bucket = s3_resource.Bucket('justatoaster-yolov5-models')
    #training_data_bucket = s3_resource.Bucket('')
    if s3_interaction:
        download_files_if_updated(args.models_bucket, 'models')
    app.run(host="0.0.0.0", port=args.port)
