"""
Program that allows you to manage a company's customer database.
The clients will be saved in a dictionary in which the key of each client will be their RFC, and the value
will be another dictionary with the client's data
(Name, Address, Telephone number, email, preferred), where preferred will have the value True if it's
about a preferred customer.

The program asks the user for an option from the following menu:

(1) Add customer,
(2) Delete customer,
(3) Show customer,
(4) List all clients,
(5) List preferred customers,
(6) Finish.

Depending on the option chosen, the program will have to do the following:

(1) Ask for the customer's data, create a dictionary with the data and add it to the database.
(2) Ask for the customer's RFC and delete customer's data from the database.
(3) Ask for the customer's RFC and show customer's data.
(4) Show list of all customer's data stored in the database.
(5) Show the list of preferred customer's data from the database.
(6) End the program.
"""


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


# Prints the menu.
def print_menu():
    print(
        "\nOption menu\n-----------\n1.- To add customer\n2.- To delete customer\n3.- To show customer\n4.- To list all customers\n5.- To list all preferred customers\n6.- Quit program\n")


# Setup database dictionaries.
customerDataBase = {}
customerRegister = {}

# Startup menu option.
menuOption = ""

# Still displaying the menu, until option "6" is selected by the user.
while menuOption != "6":

    # Add a customer.
    if menuOption == "1":
        wantToContinue = True

        while wantToContinue:
            newRFCToAdd = ""
            newCustomerRegisterToAdd = {}
            newRFCToAdd = input("RFC: ")
            customerName = input("Name: ")
            customerAddress = input("Address: ")
            customerPhoneNumber = input("Phone number: ")
            customerEmail = input("Email: ")
            isPreferredCustomer = input("Is preferred customer (Yes/No)?: ") == "Yes"
            newCustomerRegisterToAdd["name"] = customerName.upper()
            newCustomerRegisterToAdd["address"] = customerAddress.upper()
            newCustomerRegisterToAdd["phone_number"] = customerPhoneNumber
            newCustomerRegisterToAdd["email"] = customerEmail.lower()
            newCustomerRegisterToAdd["is_preferred_customer"] = isPreferredCustomer
            customerDataBase[newRFCToAdd.upper()] = newCustomerRegisterToAdd
            wantToContinue = input("Do you want to add another customer (Yes/No)? ") == "Yes"

        menuOption = ""

    # Delete a customer.
    if menuOption == "2":
        rfcToSearch = input(BColors.OKCYAN + "Please provide customer RFC: " + BColors.ENDC)

        if rfcToSearch.upper() in customerDataBase:
            del customerDataBase[rfcToSearch.upper()]
            print(BColors.WARNING + "Customer RFC", rfcToSearch.upper(), "was deleted." + BColors.ENDC)
        else:
            print(BColors.FAIL + "Customer RFC", rfcToSearch.upper(), "does not exist." + BColors.ENDC)

        menuOption = ""

    # Show a customer.
    if menuOption == "3":
        rfcToSearch = input(BColors.OKCYAN + "Please provide customer RFC: " + BColors.ENDC)

        if rfcToSearch.upper() in customerDataBase:
            customerRegister = customerDataBase[rfcToSearch.upper()]
            print(BColors.BOLD + "RFC:\t\t\t\t\t" + BColors.ENDC, rfcToSearch.upper())
            print(BColors.BOLD + "Name:\t\t\t\t\t" + BColors.ENDC, customerRegister.get("name") + ".")
            print(BColors.BOLD + "Address:\t\t\t\t" + BColors.ENDC, customerRegister.get("address") + ".")
            print(BColors.BOLD + "Phone number:\t\t\t" + BColors.ENDC, customerRegister.get("phone_number"))
            print(BColors.BOLD + "Email:\t\t\t\t\t" + BColors.ENDC, customerRegister.get("email"))
            print(BColors.BOLD + "Is preferred customer?:\t" + BColors.ENDC,
                  customerRegister.get("is_preferred_customer"))
        else:
            print(BColors.FAIL + "Customer RFC", rfcToSearch.upper(), "does not exist." + BColors.ENDC)

        menuOption = ""

    # List all customers.
    if menuOption == "4":
        print("All customer list\n-----------------\n")

        if len(customerDataBase) > 0:
            customerCounter = 0

            for rfc, customerRegister in customerDataBase.items():
                customerCounter = customerCounter + 1
                print("Customer #", customerCounter)
                print(BColors.BOLD + "RFC:\t\t\t\t\t" + BColors.ENDC, rfc)
                print(BColors.BOLD + "Name:\t\t\t\t\t" + BColors.ENDC, customerRegister.get("name") + ".")
                print(BColors.BOLD + "Address:\t\t\t\t" + BColors.ENDC, customerRegister.get("address") + ".")
                print(BColors.BOLD + "Phone number:\t\t\t" + BColors.ENDC, customerRegister.get("phone_number"))
                print(BColors.BOLD + "Email:\t\t\t\t\t" + BColors.ENDC, customerRegister.get("email"))
                print(BColors.BOLD + "Is preferred customer?:\t" + BColors.ENDC,
                      customerRegister.get("is_preferred_customer"))
                print("\n")
        else:
            print(BColors.FAIL + "The data base is empty!" + BColors.ENDC + "\n")

        menuOption = ""

    # List all preferred customers.
    if menuOption == "5":
        print("All preferred customers list\n----------------------------\n")

        customerCounter = 0

        for rfc, customerRegister in customerDataBase.items():

            if customerRegister.get("is_preferred_customer"):
                customerCounter = customerCounter + 1
                print("Customer #", customerCounter)
                print(BColors.BOLD + "RFC:\t\t\t\t\t" + BColors.ENDC, rfc)
                print(BColors.BOLD + "Name:\t\t\t\t\t" + BColors.ENDC, customerRegister.get("name") + ".")
                print(BColors.BOLD + "Address:\t\t\t\t" + BColors.ENDC, customerRegister.get("address") + ".")
                print(BColors.BOLD + "Phone number:\t\t\t" + BColors.ENDC, customerRegister.get("phone_number"))
                print(BColors.BOLD + "Email:\t\t\t\t\t" + BColors.ENDC, customerRegister.get("email"))
                print(BColors.BOLD + "Is preferred customer?:\t" + BColors.ENDC,
                      customerRegister.get("is_preferred_customer"))
                print("\n")

        if customerCounter == 0:
            print(BColors.FAIL + "There are no preferred customers in the data base!" + BColors.ENDC + "\n")

        menuOption = ""

    # Invalid menu option.
    if menuOption != "1" or menuOption != "2" or menuOption != "3" or menuOption != "4" or menuOption != "5" or menuOption != "6":
        menuOption = ""
        print_menu()
        menuOption = input(BColors.OKCYAN + "Please, select the number of an option from the menu: " + BColors.ENDC)

# Quit program.
print(BColors.OKCYAN + "Good bye!" + BColors.ENDC)
