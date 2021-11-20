function Partition(array){
    let i = -1;
    let j = i + 1;
    let len = array.length-1;
    var temp = 0
    for(k = 0; k < len+1; k++){
        if(array[j] < array[len]){
            i = i +1;
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
        j = j + 1;
    }
    temp = array[i+1];
    array[i+1] = array[len]
    array[len] = temp;

    return i+1;
}

function QuickSort(array){
    len = array.length
    if( len > 1){
        let q = Partition(array);
        let L = QuickSort(array.slice(0, q));
        let R = QuickSort(array.slice(q, len));
        let newArray = L.concat(R)
        return newArray;
    }
}

let array = [7, 4, 2, 1, 6, 8, 10, 15, 8, 3, 5]
print(QuickSort(array))