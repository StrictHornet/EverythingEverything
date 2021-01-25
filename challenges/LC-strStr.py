def strStr():
    haystack = "mississippi"

    needle = "issipi"

    # for i in range(len(haystack) - len(needle)+1):
    #     if haystack[i:i+len(needle)] == needle:
    #         return i
    # return -1
    i = 0
    j = 0

    if needle == "":
        return 0
    if len(needle) > len(haystack):
            return -1

    for letter in haystack:
        if needle[j] == letter and len(needle) <= (len(haystack)-i):
            index = i
            while(j<len(needle)-1):
                wrongString = 0
                j += 1
                i += 1
                if needle[j] == haystack[i]:
                    continue
                else:
                    print(i, j)
                    i = index
                    j = 0
                    wrongString = 1
                    break
            if wrongString != 1:
                return index
        i += 1
    return -1

print(strStr())