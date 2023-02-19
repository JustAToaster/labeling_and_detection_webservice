# labeling_and_detection_webservice
YOLOv5 labeling and detection web service with Flask and JavaScript. Created for my Master's Thesis along with the [cloud infrastructure in this repository](https://github.com/JustAToaster/CloudSystems_kops_terraform_cluster).
## Features
### Inference
The index page allows the user to choose a pre-existing YOLOv5 model, upload a JPG image and the server will be using the model for inference, then display the image for the user to see. The classes available for each model are also listed in a table, along with the number of training and validation samples the model was trained with.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33027100/219944849-452b1b76-8ef9-459a-8eb6-3b94fbbba9a3.png">
</p>

## Customizing labels
After looking at the image with the predicted bounding boxes, the user can decide to either send the predicted bounding boxes as training data to an S3 bucket or customize the bounding boxes and then send the training data.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33027100/219945080-545f5f3c-73d4-4c0c-a717-3649758ab01c.png">
</p>

After drawing a bounding box thanks to JavaScript client-side code, the user is asked with the **prompt()** function which class corresponds to that box and, after drawing all the boxes, the labels can be sent to the server by clicking the corresponding button.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33027100/219945169-eb76e9a6-76e8-4a59-a246-f3e7e273e820.png">
</p>

### Requesting models
Users can also request new models with the form in the request_model page: they need to specify the name of the model, the class list with the classes separated by new lines and optionally upload initial training data as a ZIP file.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33027100/219944976-70420ce8-5ad2-4828-9aa5-9532be6a5aba.png">
</p>

## Uploading training data for pending models
Users can upload training data for pending models requested through the form by them or another user. The pending_models page displays the classes for each pending model and the current number of training and validation samples available on the S3 bucket.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33027100/219945305-1bb075a1-0041-4a26-8cdf-4eddb1117a32.png">
</p>

after uploading the image, the user can then select the bounding boxes with their corresponding classes and send the training data, just like the pre-existing models with already available weights.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33027100/219945480-b5980bfd-7343-47cd-875c-466c0cef1ea9.png">
</p>

