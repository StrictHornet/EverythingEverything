var twoSum = function(nums, target) {
    let i = 0;
    let j = 0;
    
    for(i; i < nums.length; i++){
        j = i+1;
        //let k = j;
        for(j; j < nums.length; j++){
            if(nums[i]+nums[j]==target){
                return [i,j];
            }
        }
    }
};