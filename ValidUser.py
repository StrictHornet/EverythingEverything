# -----------------------------------------------------------
# User Data Validation
# -----------------------------------------------------------

import random
import string

#function for creating random string
def RandStr():
    letters = string.ascii_letters
    randStr = ""
    for x in range(6):
        randStr += random.choice(letters)
    return randStr

#function for collecting user password
def CreatePass():
    global userPass
    customPass = input("Kindly input custom password with length greater than 7 characters:\n")
    if len(customPass) < 7:
        print("Password should be longer than 7 characters!")
        CreatePass()
    else:
        userPass = customPass
        print("System will proceed to use custom password")

#Collecting and slicing details
lName = input("Kindly indicate your Lastname: ")
fName = input("Kindly indicate your Firstname: ")
mail = input("Kindly indicate your Email address: ")
fNameSlice = fName[0:2]
lNameSlice = lName[-2:]

#Creating Password
rand = RandStr()
userPass = fNameSlice + rand + lNameSlice
print("This is your password: ", userPass)
answer = input("Are you okay with it? Type Y for Yes and N for No:\n")
if answer == "Y":
    print("System will proceed to use auto-generated password")
elif answer == "N":
    CreatePass()
else:
    print("Type Y for Yes and N for No")

d = {"Last Name:":lName, "First Name:":fName, "E-mail:":mail, "Password:":userPass}
print("Your details are:")
for key, value in d.items():
    print(key)
    print(value)