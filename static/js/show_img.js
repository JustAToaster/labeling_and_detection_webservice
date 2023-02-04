var canvas = document.getElementById("imgCanvas");
var ctx = canvas.getContext("2d");
var send_labels_button = document.getElementById("send_labels");
var customize_labels_button = document.getElementById("customize_labels");
customize_labels_button.hidden = false;
var image;
if(document.getElementById("container_title").innerHTML == 'Uploaded image'){
  image = document.getElementById("original_image");
  send_labels_button.hidden = true;
}
else{
  image = document.getElementById("predicted_image");
  send_labels_button.hidden = false;
}
var x = 0;
var y = 0;
image.onload = function() {
  ctx.drawImage(image, x, y, canvas.width, canvas.height);
};