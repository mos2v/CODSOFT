def restaurent():
    Out_for_Delivery = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    print("Welcome to mos2v restaurent, How can I serve you today? \nNew Order\nAlready ordered")
    msg = input()
    order = []
    if msg == "New Order":
        print("What would you like to order?\nPizza\nBurger\nSweets")
        orderType = input()
        order.append(orderType)
        if orderType == "Pizza":
            print("What size would you like?\nSmall\nMedium\nLarge")
            size = input()
            order.append(size)
            print("What kind of crust would you like?\nThin\nThick")
            crust = input()
            order.append(crust)
            print("What kind of sauce would you like?\nTomato\nBBQ\nGarlic")
            sauce = input()
            order.append(sauce)
            print("What kind of cheese would you like?\nCheddar\nMozzarella\nParmesan")
            cheese = input()
            order.append(cheese)
        elif orderType == "Burger":
            print("How many patties would you like\nSingle\nDouble")
            patties = input()
            order.append(patties)
            print("Do you want to Add Bacon?\nYes\nNo")
            bacon = input()
            order.append(bacon)
            print("Do you want to Add Lettuce?\nYes\nNo")
            lettuce = input()
            order.append(lettuce)
        elif orderType == "Sweets":
            print("Which type of sweets do you want?Cake\nCookies\nIcecream")
            sweets = input()
            order.append(sweets)
            print("How many would you like?")
            quantity = input()
            order.append(quantity)
        else:
            print("Sorry we don't have that")  
    


        print("Did you finish your order?")
        ans = input()
        if ans == "Yes":
            print("Do you want your order to be Delivery or Pickup?")
            ans2 = input()
        if ans2 == "Delivery":
            print("Where would you like your order to be delivered?")
            ans3 = input()
            order.append(ans3)
            print("Here is all your order details")
            for i in order:
                print(i)
            print("Thank you for your order")       
    elif msg == "Already ordered":
        print("Please provide your order number")
        orderNum = input()
        for i in Out_for_Delivery:
            if orderNum == i:
                print("Your Order is Out for Delivery")
                return
        print("Your Order is being Prepeared")    
def chatbot():
    while True:
        restaurent()


chatbot()           