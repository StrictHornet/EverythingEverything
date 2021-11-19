array = [7, 4, 2, 1, 6, 8, 10, 15, 8, 3, 5]

def Partition(array):
    p = len(array)-1
    i = -1
    j = i + 1
    for element in array:
        if array[j] < array[p]:
            i += 1
            temp = array[i]
            array[i] = array[j]
            array[j] = temp
        j += 1
        print(array)
    temp = array[i+1]
    array[i+1] = array[p]
    array[p] = temp
    return array

print(Partition(array))