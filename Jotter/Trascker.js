//REVERSE SRTING FUNCTION (10 DAYS OF JS)
function reverseString(s) {
    try{
        var split_s = s.split("");
        var concat_s = "";
        for(let x of split_s){
            concat_s = x.concat(concat_s);
    }
        s = concat_s;
    }
    catch(e){
        console.log(e.message);
    }
    console.log(s);
}

//10-03-2021
//Multiple Linear regressor on google colabs

/*
 * Complete the isPositive function.
 * If 'a' is positive, return "YES".
 * If 'a' is 0, throw an Error with the message "Zero Error"
 * If 'a' is negative, throw an Error with the message "Negative Error"
 */
function isPositive(a) {
    if(a>0){
        return "YES"
    }
    else if(a == 0){
        throw new Error("Zero Error");
    }
    else{
        throw new Error("Negative Error")
    }
}


/*
 * Complete the Rectangle function
 */
function Rectangle(a, b) {
    var Quad = {
        length : a,
        width : b,
        perimeter : 2 * (a+b), 
        area : a * b
    };
    
    return Quad
}

/*
 * Return a count of the total number of objects 'o' satisfying o.x == o.y.
 * 
 * Parameter(s):
 * objects: an array of objects with integer properties 'x' and 'y'
 */
function getCount(objects) {
    var count = 0;
    for(let object of objects){
        if(object.x == object.y){
            count++;
        }
    }
    return count;
}

/*
 * Implement a Polygon class with the following properties:
 * 1. A constructor that takes an array of integer side lengths.
 * 2. A 'perimeter' method that returns the sum of the Polygon's side lengths.
 */

class Polygon{
    constructor(lengths){
        this.list = lengths;
    }
    
    perimeter(){
        var sum = 0;
        var arr = this.list;
        for(let element of arr){
            sum = sum + element;
        }
        
        return sum;
    }
}

class Rectangle {
    constructor(w, h) {
        this.w = w;
        this.h = h;
    }
}

/*
 *  Write code that adds an 'area' method to the Rectangle class' prototype
 */
Rectangle.prototype.area = function(){
    return this.w * this.h;
}
/*
 * Create a Square class that inherits from Rectangle and implement its class constructor
 */
class Square extends Rectangle {
    constructor(w) {
        super(w);
        this.h = w
        super.area();
    }
}

/*
Convert image to grayscale
void grayscale(int height, int width, RGBTRIPLE image[height][width])
{
    int average;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            average = (int) round((image[i][j].rgbtBlue + image[i][j].rgbtGreen + image[i][j].rgbtRed) / 3);
            image[i][j].rgbtBlue = average;
            image[i][j].rgbtGreen = average;
            image[i][j].rgbtRed = average;
            
            printf("%u", image[i][j].rgbtBlue);
            printf("|||");
            printf("%x", image[i][j].rgbtRed);
            printf(".\n");
        }
    }
    return;
}
*/

