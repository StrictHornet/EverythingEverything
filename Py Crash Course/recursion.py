def main():
    # prints pyramid
    pyramid(6)


def pyramid(n):
    if n == 1:
        block = "#"
        print(block)
    else:
        block = pyramid(n-1) + "#"
        print(block)
        
    return block

if __name__ == "__main__":
    main()