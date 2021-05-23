setInterval(Red,1000.0);
setInterval(Green,2000);
// 



function Green(){
    spacebar = document.querySelector("#spacebar")
    spacebar.style.backgroundColor = "Green"
}

function Red(){
    spacebar = document.querySelector("#spacebar")
    spacebar.style.backgroundColor = "Red"
    
}

// function Jumbotron(){
//     spacebar = document.querySelector("#spacebar")
//     spacebar.style.backgroundColor = "Green"

//     spacebar.removeEventListener("click", Jumbotron)
//     document.querySelector("#spacebar").addEventListener("click", function FlipFunction(){
//          document.querySelector("#spacebar").style.backgroundColor = "Red"
//          document.querySelector("#spacebar").removeEventListener("click", FlipFunction)
//          document.querySelector("#spacebar").addEventListener("click", Jumbotron)
//     })
// }

// function listen(){
//     // document.querySelector(".container").addEventListener("mouseover", changeColor)
//     // document.querySelector(".container").addEventListener("mouseover", povertyCapital)
//     document.querySelector("#spacebar").addEventListener("click", Jumbotron)
// }

// document.addEventListener("DOMContentLoaded", listen)