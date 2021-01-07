#objective is to stay within budget
keyboards = [12, 40, 20, 30, 45, 65, 700, 300, 90]
drives = [4, 6, 10, 400, 350, 270, 930, 23, 45, 78, 198, 162]
b = 345

top = 0
keyboards.sort(reverse = True)
drives.sort()
for kb in keyboards:
    if kb > b:
        continue
    for dv in drives:
        if dv > b:
            continue
        if kb + dv > b:
            break
        if kb + dv > top:
            top = kb + dv
        

if top == 0:
    print("No Possible Combination")
else:
    print(top)