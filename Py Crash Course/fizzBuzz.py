# divisible by 3 return fizz
# divisible by 5 return buzz
# else return number


def FizzBuzz(num):
    if num % 3 == 0 and num % 5 == 0:
        return "fizz buzz"
    elif num % 3 == 0:
        return "fizz"
    elif num % 5 == 0:
        return "buzz"
    else:
        return num


print(FizzBuzz(15))
