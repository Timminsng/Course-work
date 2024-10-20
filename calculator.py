previous_answer = None

def calculator(expression):
    """
    TODO: Write a function that takes an equation as a string and returns the answer. 
    The equation will be in the form of a string with the following operators: +, -, *, /, and ^.
    The equation could contain the string "ans" which will be replaced with the previous answer.
    If the equation is invalid, print an error message and return "null".
    The addition of helper functions is suggested but not required.
    @param  expression --> a string that represents an equation
    @return result --> the answer to the equation
    """
    global previous_answer
    try:
        if previous_answer is not None:
            expression = expression.replace("ans", str(previous_answer))
        expression = expression.replace("^", "**")
        result = eval(expression)
        previous_answer = result
        return result
    except Exception as e:
        print("Invalid Operation")
        return "null"

# It is unnecessary to edit the "main" function of each problem's provided code skeleton.
# The main function is written for you in order to help you conform to input and output formatting requirements.
def main():
    expression = input("Equation: ")
    while expression != "exit":
        answer = calculator(expression)
        print("= ", answer)
        expression = input("Enter an expression: ")


main()