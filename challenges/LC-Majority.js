function Majority(nums){
    //console.log(nums)
    var arrlen = nums.length;
    if (arrlen == 0) return 0;
    if (arrlen == 1) return nums[0];
    
    var half = Math.ceil(arrlen/2);
    var arrtemp = [];
    
    for(let i = 0; i <= half; i+=2){
        if (nums[i+1] == NaN) arrtemp.push(nums[i]);
        if (nums[i] == nums[i+1]) arrtemp.push(nums[i]);
        //console.log(nums[i], nums[i+1]);
    }
    console.log(arrtemp)
    return Majority(arrtemp);
}

var majorityElement = function(nums) {
    let result = Majority(nums);
    console.log(result);
    let count = 0;
    
    for(let i = 0; i <= nums.length; i++){
        if (nums[i] == result) count++;
    }
    
    if (count > nums.length/2) return result;
    
};