function post(data){
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/data", true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.send(JSON.stringify(data));
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
    post(data)
}


setInterval(()=>{
    main()
    // post(data)
}, 100);