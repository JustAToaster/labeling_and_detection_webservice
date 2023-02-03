var boundingBoxes = [];

function reloadAsGet()
{
    var loc = window.location;
    window.location = loc.protocol + '//' + loc.host + loc.pathname + loc.search;
}

let send_labels = () => {
  //Label transfer to server
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/save-labels", true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onreadystatechange = function() {
    if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
      console.log("Bounding boxes saved successfully");
    }
  };
  var img_name = document.getElementById("original_image").src.split("/").pop();
  var model_index = document.getElementById("model_index").innerHTML;
  xhr.send(JSON.stringify({model_index, img_name, boundingBoxes}));

  alert("Labels sent to the server!");
  reloadAsGet();
}

let customize_labels = () => {
  var container_title = document.getElementById("container_title");
  container_title.innerHTML = "Labeling image";
  //var send_labels_button = document.getElementById("send_labels");
  var customize_labels_button = document.getElementById("customize_labels");
  //send_labels_button.hidden = true;
  customize_labels_button.hidden = true;

  var canvas = document.getElementById("imgCanvas");
  var ctx = canvas.getContext("2d");
  var image = document.getElementById("original_image");
  var width = 400;
  var height = 400;
  var x = 0;
  var y = 0;
  ctx.drawImage(image, x, y, width, height);

  var max_boundingBoxes = 16;

  var startX, startY, endX, endY;
  var isDrawing = false;

  canvas.addEventListener("mousedown", function(event) {
    if (boundingBoxes.length >= max_boundingBoxes) return;
    isDrawing = true;
    startX = event.clientX - canvas.offsetLeft;
    startY = event.clientY - canvas.offsetTop;
  });

  canvas.addEventListener("mouseup", function(event) {
    if (boundingBoxes.length >= max_boundingBoxes) return;
    isDrawing = false;
    endX = event.clientX - canvas.offsetLeft;
    endY = event.clientY - canvas.offsetTop;
    drawRectangle();
    saveBoundingBox();
  });

  canvas.addEventListener("mousemove", function(event) {
    if (boundingBoxes.length >= max_boundingBoxes) return;
    if (isDrawing) {
      endX = event.clientX - canvas.offsetLeft;
      endY = event.clientY - canvas.offsetTop;
      drawRectangle();
    }
  });

  function drawBoundingBoxes() {
    for (let i = 0; i < boundingBoxes.length; i++) {
      curr_box = boundingBoxes[i];
      ctx.beginPath();
      ctx.strokeRect(curr_box['startX'], curr_box['startY'], curr_box['endX'] - curr_box['startX'], curr_box['endY'] - curr_box['startY']);
    }
  }

  function saveBoundingBox() {
    var centerX = (startX + endX) / 2 / canvas.width;
    var centerY = (startY + endY) / 2 / canvas.height;
    var boxWidth = (endX - startX) / canvas.width;
    var boxHeight = (endY - startY) / canvas.height;
    var class_index = prompt("What class is in this bounding box? Insert the number corresponding to the class.", "0");
    boundingBoxes.push({
      startX: startX,
      startY: startY,
      endX: endX,
      endY: endY,
      centerX: centerX,
      centerY: centerY,
      boxWidth: boxWidth,
      boxHeight: boxHeight,
      class_index: class_index
    });
    console.log("Saved bounding box:", boundingBoxes[boundingBoxes.length - 1]);
  }

  function drawRectangle() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    drawBoundingBoxes();
    ctx.beginPath();
    ctx.strokeRect(startX, startY, endX - startX, endY - startY);
  }

}