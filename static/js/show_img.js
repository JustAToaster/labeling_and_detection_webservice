var canvas = document.getElementById("imgCanvas");
var ctx = canvas.getContext("2d");
var image = document.getElementById("predicted_image");
var x = 0;
var y = 0;
image.onload = function() {
  ctx.drawImage(image, x, y, canvas.width, canvas.height);
};
var send_labels_button = document.getElementById("send_labels");
var customize_labels_button = document.getElementById("customize_labels");
send_labels_button.hidden = false;
customize_labels_button.hidden = false;