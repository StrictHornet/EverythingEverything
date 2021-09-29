var v1, v2, res;
var ops = "";

function clickOne(){
    document.querySelector("#res").innerHTML += "1";
}

function clickZero(){
    document.querySelector("#res").innerHTML += "0";
}

function clickSum(){
    v1 = document.querySelector("#res").innerHTML;
    document.querySelector("#res").innerHTML = "";
    ops = "+"
}

function clickEql(){
    v2 = document.querySelector("#res").innerHTML;
    
    switch (ops) {
        case "+":
            res = v1 + v2;
            break;
    
        default:
            res = "Err"
            break;
    }
    document.querySelector("#res").innerHTML = res;
}