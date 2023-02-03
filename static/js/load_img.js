$('#inputfile').bind('change', function() {
    let fileSize = this.files[0].size/1024/1024;
    if (fileSize > 1) {
      $("#inputfile").val(null);
      alert('Image should be 1MB or smaller')
      return
    }

    let ext = $('#inputfile').val().split('.').pop().toLowerCase();
    if($.inArray(ext, ['jpg','jpeg']) == -1) {
      $("#inputfile").val(null);
      alert('This is not a JPG file.');
    }
});