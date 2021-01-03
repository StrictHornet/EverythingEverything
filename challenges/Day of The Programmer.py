def DOP():    
    uInput = int(input("What year is it? :- "))
    print(uInput)
    if uInput == 1918:
        print(f"26.09.{uInput}")
    elif (uInput < 1918 and uInput % 4 == 0) or ((uInput > 1918) and (uInput % 400 == 0 or (uInput % 4 == 0 and uInput % 100 != 0))):
        print(f"12.09.{uInput}")
    else:
        print(f"13.09.{uInput}")
        
DOP()