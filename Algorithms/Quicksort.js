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
    console.log(array)
    return i+1;
}

function QuickSort(array){
    //console.log(array)
    len = array.length
    if(len < 2){
        return array;
    }
    if( len > 1){
        let q = Partition(array);
        if(q==0){
            q = q+1;
        }
        console.log(q, array[q])
        let L = array.slice(0, q)
        console.log(L)
        let R = array.slice(q, len)
        console.log(R)
        let LQS = QuickSort(L);
        let RQS = QuickSort(R);
        console.log(LQS, RQS)
        return LQS.concat(RQS);
    }
}

let array = [7, 4, 2, 1, 6, 8, 10, 15, 3, 5]
console.log(QuickSort(array))