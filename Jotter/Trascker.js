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

// Convert image to sepia
// void sepia(int height, int width, RGBTRIPLE image[height][width])
// {
//     for(int i = 0; i < height; i++){
//         for(int j = 0; j < width; j++){
//             image[i][j].rgbtBlue = (int) round((0.131 * image[i][j].rgbtBlue) + (0.534 * image[i][j].rgbtGreen) + (0.272 * image[i][j].rgbtRed));
//             image[i][j].rgbtGreen = (int) round((0.168 * image[i][j].rgbtBlue) + (0.686 * image[i][j].rgbtGreen) + (0.349 * image[i][j].rgbtRed));
//             image[i][j].rgbtRed = (int) round((0.189 * image[i][j].rgbtBlue) + (0.769 * image[i][j].rgbtGreen) + (0.393 * image[i][j].rgbtRed));
            
//             if(image[i][j].rgbtBlue > 255){
//                 image[i][j].rgbtBlue = 255;
//             }
//             if(image[i][j].rgbtGreen > 255){
//                 image[i][j].rgbtGreen = 255;
//             }
//             if(image[i][j].rgbtRed > 255){
//                 image[i][j].rgbtRed = 255;
//             }
//         }
//     }
//     return;
// }

// Convert image to sepia 2,0
// void sepia(int height, int width, RGBTRIPLE image[height][width])
// {

//     RGBTRIPLE temp[height][len];
//     for(int i = 0; i < height; i++){
//         for(int j = 0; j < len; j++){
//             temp[i][j] = image[i][j];
//         }


//     for(int i = 0; i < height; i++){
//         for(int j = 0; j < width; j++){
//             image[i][j].rgbtBlue = (int) round((0.131 * temp[i][j].rgbtBlue) + (0.534 * temp[i][j].rgbtGreen) + (0.272 * temp[i][j].rgbtRed));
//             image[i][j].rgbtGreen = (int) round((0.168 * temp[i][j].rgbtBlue) + (0.686 * temp[i][j].rgbtGreen) + (0.349 * temp[i][j].rgbtRed));
//             image[i][j].rgbtRed = (int) round((0.189 * temp[i][j].rgbtBlue) + (0.769 * temp[i][j].rgbtGreen) + (0.393 * temp[i][j].rgbtRed));
            
//             if(image[i][j].rgbtBlue > 255){
//                 image[i][j].rgbtBlue = 255;
//             }
//             if(image[i][j].rgbtGreen > 255){
//                 image[i][j].rgbtGreen = 255;
//             }
//             if(image[i][j].rgbtRed > 255){
//                 image[i][j].rgbtRed = 255;
//             }
//         }
//     }
//     return;
// }
// Unloads dictionary from memory, returning true if successful, else false
// bool unload(void)
// {
//     // TODO
//     for(int i = 0; i < 26; i++)
//     {
//         node *list = table[i];

//         while(list!= NULL)
//         {
//             node *tmp = list;
//             list = list->next;
//             free(tmp);
//         }

//         //free(list);

//         if(i == 25 && list == NULL)
//         {
//             printf("FRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR");
//             return true;
//         }
//     }
//     return false;
// }
//WHAT FREE DOES, WHAT IT FREES AND WHY FREE(LIST) DID/DOES NOTHING
//SKETCH LINKED LIST TO FIND LEAK\

// //cs50 lab6
//     # TODO: Read teams into memory from file
//     teams = []

//     with open(sys.argv[1], "r") as file:
//         teams_dict = csv.DictReader(f)
//         for t in teams_dict:
//             team = {"team" : t["team"], "rating" : int(t["rating"])}
//             teams.append(team)

//     counts = {}
//     # TODO: Simulate N tournaments and keep track of win counts
//     for i in range(N):
//         winner = simulate_tournament(teams)
//         #print(winner)
//         if winner in counts:
//             #add 1
//             counts[winner] = counts[winner] + 1
//         else:
//             counts[winner] = 1



// def simulate_tournament(teams):
//     """Simulate a tournament. Return name of winning team."""
//     # TODO
//     rounds = int(math.sqrt(len(teams)))
//     for i in range(rounds):
//         teams = simulate_round(teams)

//     return teams[0]["team"]

// DNA.PY
// import sys
// import csv

// # Function to find STR's
// def StrFinder(sequence, str_types):
//     sequence_length = len(sequence)
//     str_finder_found = {}
//     for str_type in str_types:
//         str_finder_found[str_type] = 0
//         i = 0
//         length_str_type = len(str_type)

//         # Loop sequence to find STR
//         while i < sequence_length:
//             if sequence[i: i+length_str_type] == str_type:
//                 count = 0
//                 while sequence[i: i+length_str_type] == str_type:
//                     count += 1
//                     i += length_str_type
//                 if count > str_finder_found[str_type]:
//                     str_finder_found[str_type] = count
//             else:
//                 i += 1

//     #print(str_finder_found)
//     return str_finder_found

// # Defining Variables
// str_counts = []
// sequence = ""
// str_types = []
// found = False

// # Check argument length
// if len(sys.argv) != 3:
//     print("Incorrect amount of arguments")
//     sys.exit(1)

// # Import CSV and TXT
// with open(sys.argv[1], "r") as str_counts_csv:
//     reader = csv.DictReader(str_counts_csv)

//     for row in reader:
//         str_counts.append(row)

// with open(sys.argv[2], "r") as sequence_txt:
//     for line in sequence_txt:
//         sequence = line

// # Get Str's to be looked for
// str_types_count = len(str_counts[0])
// for str_type in [*str_counts[0].keys()]:
//     if str_type != "name":
//         str_types.append(str_type)
//     #print(str_types)
        
// # Find Str's in sequence
// str_found = StrFinder(sequence, str_types)
// #print(str_found)

// # Compare STR FOUND WITH CSV
// for str_count in str_counts:
//     found = []
//     for str_type in str_types:
//         #print(str_count[str_type], str_found[str_type])
//         if int(str_count[str_type]) == int(str_found[str_type]):
//             found.append(True)
//         else:
//             found.append(False)
//     if False not in found:
//         print(str_count["name"])
//         break

// if False in found:
//     print(str_count["name"])
        

// ##################################################################################################################
// # for str_count in str_counts:
// #     print(str_count["name"], str_count["TATC"])
// #print(sequence, end ="")
// #print(str_types)
//check50 cs50/problems/2021/x/dna

// LAB 7 SQL
// SELECT name FROM songs WHERE name LIKE "%feat%";
// SELECT AVG(energy) FROM songs WHERE artist_id IN(SELECT id FROM artists WHERE name = "Drake");
// SELECT name FROM songs WHERE artist_id IN(SELECT id FROM artists WHERE name = "Post Malone");
// SELECT AVG(energy) FROM songs;
// SELECT name FROM songs WHERE danceability > 0.75 AND energy > 0.75 AND valence > 0.75;
// SELECT name FROM songs
// ORDER BY duration_ms
// DESC LIMIT 5;
// SELECT name FROM songs
// ORDER BY tempo;
// SELECT name FROM songs;


// -- See Courthouse Reports
// SELECT activity, license_plate, hour FROM courthouse_security_logs
// WHERE year = 2020 AND month = 7 AND day = 28 AND hour =10
// AND minute < 30;

// -- Plates that left
// -- exit | 5P2BI95 | 10
// -- exit | 94KL13X | 10
// -- exit | 6P58WS2 | 10
// -- exit | 4328GD8 | 10
// -- exit | G412CB7 | 10
// -- exit | L93JTIZ | 10CS50
// -- exit | 322W7JE | 10
// -- exit | 0NTHK55 | 10

// -- See Flight