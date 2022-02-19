function initialise(person){
    let name = person.toUpperCase();        //To ensure output is capital alphabets
    let init = "";
    let fnIndex = 0;                        //Variable that stores the index to be used for first initial

    //First checks if there is a number at any point during input traversal,
    //then adds the first initial to the init variable, followed by a check
    //to ensure that the input name is an actual Fullname not just a firstname
    //but a firstname with a lastname that is more than a letter
    for(var i = 0; i < name.length; i++){
        if(typeof name[i] == "number"){
            return "Input names without numbers";
        }
        
        if(i == fnIndex){
            init += name[i] + ".";
        }
        
        if(name.charCodeAt(i) == 32){
            if(name.length - (i+1) < 2){
                return "Input fullname";
            }
            else{
                init += name[i+1];
                return init;
            }
        }
    }
}

console.log(initialise("Isabella Quinn"));