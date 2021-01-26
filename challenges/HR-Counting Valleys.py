path = "UDDDUDUUDDUU"
start = 0
count = 0
for element in path:
    start += 1 if element == "U" else -1
    if element == "U" and start == 0:
        count += 1
print(count)