/*
 * Complete the 'jumpingOnClouds' function below.
 *
 * The function is expected to return an INTEGER.
 * The function accepts INTEGER_ARRAY c as parameter.
 */

function jumpingOnClouds(c) {
    // Write your code here
    var arr = c;
    var count = 0;
    var min = 0;
    var len = arr.length;
    var i = 0;
    var num;
    for (num in c)  {
        if (i >= len || i == (len -1)){
            return count;
        }
        breakme:
        if (arr[i] == 0){
            if (arr[i+1] == 0 && arr[i+2] == 0){
                i = i + 2
                count+=1;
                break breakme;
            }
            else if (arr[i+1] == 0 && arr[i+2] == 1){
                i = i + 1;
                count+=1;
                break breakme;
            }
            else{
            i = i + 2;
            count+=1;  
            }
        }
        else{
         return "na wa oo";
        }
        //min = len - count;
    }
    //i += 1;
    return count;
}