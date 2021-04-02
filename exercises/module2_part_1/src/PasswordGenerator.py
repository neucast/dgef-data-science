import string
import random

LETTERS = string.ascii_letters
NUMBERS = string.digits
SYMBOLS = string.punctuation


# Inner class to statically store terminal colors.
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


"""
Asks for the length of the password.
:returns number <class 'int'>
"""


def get_password_length():
    length = input("How long do you want your password? ")
    return int(length)


"""
Generates a random password having the specified length.
:length -> length of password to be generated. Defaults to 8 chars if nothing is specified.
:passwordCombinationChoiceList-> a list of boolean values representing a user choice for
    string constant to be used to generate password.
    0 item ---> digits
         True to add digits to constant False otherwise.
    1 item ---> letters
         True to add letters to constant False otherwise.
    2 item ---> symbols
         True to add symbols to constant False otherwise.
:returns string <class 'str'>
"""


def password_generator(passwordCombinationChoiceList, length=16):
    # create alphanumerical by fetching string constant
    printable = fetch_string_constant(passwordCombinationChoiceList)

    # convert printable from string to list and shuffle
    printable = list(printable)
    random.shuffle(printable)

    # generate random password
    random_password = random.choices(printable, k=length)

    # convert generated password to string
    random_password = "".join(random_password)
    return random_password


"""
Returns a string constant based on users passwordCombinationChoiceList.
string constant can either be digits, letters, symbols or
combination of them.
: passwordCombinationChoiceList --> list <class 'list'> of boolean
    0 item ---> digits
        True to add digits to constant False otherwise
    1 item ---> letters
        True to add letters to constant False otherwise
    2 item ---> symbols
        True to add symbols to constant False otherwise
"""


def fetch_string_constant(passwordCombinationChoiceList):
    string_constant = ""

    string_constant += NUMBERS if passwordCombinationChoiceList[0] else ""
    string_constant += LETTERS if passwordCombinationChoiceList[1] else ""
    string_constant += SYMBOLS if passwordCombinationChoiceList[2] else ""

    return string_constant


"""
Sets the agreement of the secure combination of characters for a password.
:returns list <class 'list'> of boolean [True, True, True]
    0 item ---> digits
    1 item ---> letters
    2 item ---> symbols
"""


def secure_password_combination_choice():
    return [True, True, True]


"""
Prompt a user to choose password character combination which
could either be digits, letters, symbols or combination of
either of them.
:returns list <class 'list'> of boolean [True, True, False]
    0 item ---> digits
    1 item ---> letters
    2 item ---> symbols
"""


def password_combination_choice():
    # Retrieve a user's password character combination choice.
    wantDigits = input("Want digits  (Yes/No)? ").upper() == "YES"
    wantLetters = input("Want letters  (Yes/No)? ").upper() == "YES"
    wantSymbols = input("Want symbols (Yes/No)? ").upper() == "YES"

    return [wantDigits, wantLetters, wantSymbols]


if __name__ == '__main__':

    wantsAnotherPassword = True

    while wantsAnotherPassword:
        length = get_password_length()
        passwordCombinationChoiceList = secure_password_combination_choice()
        password = password_generator(passwordCombinationChoiceList, length)

        print(BColors.OKBLUE + "The generated password is:" + BColors.ENDC, password)

        wantsAnotherPassword = input("Do you want to generate another password (Yes/No)? ").upper() == "YES"
