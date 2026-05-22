class Basic_Arithmetic:

    @staticmethod
    def add(number_list):
        total = 0
        for number in number_list:
            total += number
        return total

    @staticmethod
    def sub(number_list):
        total = number_list[0]
        for number in number_list[1:]:
            total -= number
        return total

    @staticmethod
    def mul(number_list):
        total = 1
        for number in number_list:
            total *= number
        return total

    @staticmethod
    def div(number_list):
        total = number_list[0]
        for number in number_list[1:]:
            total /= number
        return total

    @staticmethod
    def exp(number_list):
        total = number_list[0]
        for number in number_list[1:]:
            total **= number
        return total

class Advanced_Arithmetic:

    @staticmethod
    def factorial(number):
        total = 1
        for i in range(1, number + 1):
            total *= i
        return total
    
    @staticmethod
    def fractional_factorial(number):
        pi = 3.14159
        return number * (pi ** number)
    
    @staticmethod
    def power(number, exponent):
        return number ** exponent
    
    @staticmethod
    def sqrt(number):
        return number ** (1 / 2)
    
    @staticmethod
    def curt(number):
        return number ** (1 / 3)
    
    @staticmethod
    def sq(number):
        return number ** 2
    
    @staticmethod
    def cub(number):
        return number ** 3
    
    @staticmethod
    def permutations(obj, number):
        result1 = 1
        result2 = 1
        dif = number - obj
        for i in range(1, number + 1):
            result1 *= i
        for i in range(1, dif + 1):
            result2 *= i
        return result1 / result2
    
    @staticmethod
    def combinations(obj, number):
        result1 = 1
        result2 = 1
        result3 = 1
        dif = number - obj
        for i in range(1, number + 1):
            result1 *= i
        for i in range(1, dif + 1):
            result2 *= i
        for i in range(1, obj + 1):
            result3 *= i
        return result1 / (result3*result2)