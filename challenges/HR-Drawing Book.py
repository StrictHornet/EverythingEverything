print("Enter number of pages and page number to get to.")
n = int(input("Enter number of pages: "))
p = int(input("Enter page number: "))
front = 1
back = n
fcount = 0
bcount = 0
minFlips = 0

while(front<p):
    front += 2
    fcount += 1

if n%2==0:
    while(back>p):
        back -= 2
        bcount += 1
else:
    while(back>(p+1)):
        back -= 2
        bcount += 1

minFlips = fcount if fcount < bcount else bcount

print(minFlips)