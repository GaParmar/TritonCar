function post_log_path(data){
    var rpi_url = document.getElementById("rpi_url").value
    var xhr = new XMLHttpRequest();
    xhr.open("POST", rpi_url + "/log_path?log_path=" + data, true);
    xhr.send();
}

function get_log_path(callback){
    var rpi_url = document.getElementById("rpi_url").value
    var xhr = new XMLHttpRequest();

    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            callback(xhr.responseText)
        } 
  
        //it failed to respond correctly, gives an error message and tells user to check console
        else if(xhr.readyState === 4){
            console.log("FAILED")
            console.log(xhr.status)
        }
    };
    xhr.open("GET", rpi_url + "/log_path", true);
    xhr.send();
}