
WIDTH = 7
HEIGHT = 7

directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]
def main():
    for index in range(WIDTH * HEIGHT): 
        y = index // WIDTH
        x = index - y * WIDTH
        count = 0
        for dx, dy in directions:
            _x = x + dx
            _y = y + dy
            
            if 0 <= _x < WIDTH  and 0 <= _y < HEIGHT:
                count += 1
        print(count, end=' ')   

        if (index+1) % HEIGHT == 0:
            print('')
    

if __name__ == '__main__':
    main()