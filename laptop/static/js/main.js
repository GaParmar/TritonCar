function post(data){
    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/data", true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.onreadystatechange = function(){
        if (xhr.readyState == XMLHttpRequest.DONE && xhr.response=="failure"){
            document.getElementById("p_status").innerHTML = "disconnected"
            document.getElementById("p_status").style.color = "red"
        } 
    }
    xhr.send(JSON.stringify(data));
}

function update_ui(data){
    document.getElementById("p_throttle").innerHTML = data["ly"]
    document.getElementById("p_steer").innerHTML = data["rx"]

}

function socket_connect(){
    let port = document.getElementById("inp_port").value
    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/init_connection", true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            let res = JSON.parse(xhr.response);
            // update the ui with the response
            if (res.status == "success"){
                document.getElementById("p_status").innerHTML = "connected"
                document.getElementById("p_status").style.color = "green"
            }
        }
    }
    xhr.send(JSON.stringify({"port":port}));
}

function main(){
    let controller = null;
    let gamepads = navigator.getGamepads();
    if(gamepads.length > 0){
        for(let i = 0; i<gamepads.length; i+=1){
          if (gamepads[i] === null){}
          else{controller = gamepads[i]}
        }
    }
    if (controller==null){return}

    // range[-1,1] to [0,1] to [0,255]
    data = {
        "ly"        : parseInt((controller.axes[1]*0.5 + 0.5)*255),
        "rx"        : parseInt((controller.axes[2]*0.5 + 0.5)*255),
        "square"    : controller.buttons[2].pressed,
        "triangle"  : controller.buttons[3].pressed,
        "circle"    : controller.buttons[1].pressed,
        "cross"     : controller.buttons[0].pressed
    }

    if (document.getElementById("p_status").innerHTML == "connected"){
        // send data to the flask server
        post(data)
    }
    // update the ui
    update_ui(data)
}


setInterval(()=>{
    main()
}, 150);

window.onload = function(e){ 
    // get local host ip address on page load
    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/get_host_ip", true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            document.getElementById("p_ip").innerHTML = xhr.response
        }
    }
    xhr.send(JSON.stringify({}));
}