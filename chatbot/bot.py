def restaurent():
    Out_for_Delivery = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    print("Welcome to mos2v restaurent, How can I serve you today? \nNew Order\nAlready ordered")
    msg = input().strip()
    order = []
    if msg.lower() == "new order":
        print("What would you like to order?\nPizza\nBurger\nSweets")
        orderType = input().strip()
        order.append(orderType)
        if orderType.lower() == "pizza":
            print("What size would you like?\nSmall\nMedium\nLarge")
            size = input().strip()
            order.append(size)
            print("What kind of crust would you like?\nThin\nThick")
            crust = input().strip()
            order.append(crust)
            print("What kind of sauce would you like?\nTomato\nBBQ\nGarlic")
            sauce = input().strip()
            order.append(sauce)
            print("What kind of cheese would you like?\nCheddar\nMozzarella\nParmesan")
            cheese = input().strip()
            order.append(cheese)
        elif orderType.lower() == "burger":
            print("How many patties would you like\nSingle\nDouble")
            patties = input().strip()
            order.append(patties)
            print("Do you want to Add Bacon?\nYes\nNo")
            bacon = input().strip()
            order.append(bacon)
            print("Do you want to Add Lettuce?\nYes\nNo")
            lettuce = input().strip()
            order.append(lettuce)
        elif orderType.lower() == "sweets":
            print("Which type of sweets do you want?Cake\nCookies\nIcecream")
            sweets = input().strip()
            order.append(sweets)
            print("How many would you like?")
            quantity = input().strip()
            order.append(quantity)
        else:
            print("Sorry we don't have that")  
            return


        print("Did you finish your order? (Yes/No)")
        ans = input().strip()
        if ans.lower() == "yes":
            print("Do you want your order to be Delivery or Pickup?")
            ans2 = input().strip()
            if ans2.lower() == "delivery":
                print("Where would you like your order to be delivered?")
                ans3 = input().strip()
                order.append(ans3)
            
            print("Here is all your order details")
            for i in order:
                print(i)
            print("Thank you for your order")       
    
    elif msg.lower() == "already ordered":
        print("Please provide your order number")
        orderNum = input().strip()
        if orderNum.isdigit() and int(orderNum) in Out_for_Delivery:
            print("Your Order is Out for Delivery")
        else:
            print("Your Order is being Prepeared") 
    else:   
        print("Sorry, I didn't understand that. Please choose 'New Order' or 'Already ordered'.")


def chatbot():
    while True:
        restaurent()
        print("Would you like to do anything else? (Yes/No)")
        response = input().strip()
        if response.lower() != "yes":
            print("Goodbye!")
            break


chatbot()           