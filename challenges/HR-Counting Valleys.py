path = "UDDDUDUUDDUU"
start = 0
count = 0
for element in path:
    if element == "U":
        start += 1
    else:
        start += -1
    if element == "U" and start == 0:
        count += 1
print(count)