#!/usr/bin/env python
# coding: utf-8

# In[1]:


#11. Write a python program to find the factorial of a number.
#To take input from the user
num = int(input("Enter a number: "))

factorial = 1

# check if the number is negative, positive or zero
if num < 0:
   print("Sorry, factorial does not exist for negative numbers")
elif num == 0:
   print("The factorial of 0 is 1")
else:
   for i in range(1,num + 1):
       factorial = factorial*i
   print("The factorial of",num,"is",factorial)


# In[2]:


#12. Write a python program to find whether a number is prime or composite.
num = int(input("Enter any number : "))
if num > 1:
    for i in range(2, num):
        if (num % i) == 0:
            print(num, "is NOT a prime number")
            break
    else:
        print(num, "is a PRIME number")
elif num == 0 or 1:
    print(num, "is a neither prime NOR composite number")
else:
    print(num, "is NOT a prime number it is a COMPOSITE number")


# In[3]:


#13. Write a python program to check whether a given string is palindrome or not.
input_str = input("Enter a string: ")

final_str = ""
rev = reversed(input_str)

if list(input_str) == list(rev):
    print(input_str, "is palindrome")
else:
    print(input_str, "is not palindrome")


# In[4]:


#14. Write a Python program to get the third side of right-angled triangle from two given sides.
def pythagoras(opposite_side,adjacent_side,hypotenuse):
        if opposite_side == str("x"):
            return ("Opposite = " + str(((hypotenuse**2) - (adjacent_side**2))**0.5))
        elif adjacent_side == str("x"):
            return ("Adjacent = " + str(((hypotenuse**2) - (opposite_side**2))**0.5))
        elif hypotenuse == str("x"):
            return ("Hypotenuse = " + str(((opposite_side**2) + (adjacent_side**2))**0.5))
        else:
            return "You know the answer!"
    
print(pythagoras(3,4,'x'))
print(pythagoras(3,'x',5))
print(pythagoras('x',4,5))
print(pythagoras(3,4,5))


# In[5]:


#15. Write a python program to print the frequency of each of the characters present in a given string.
string=input("Enter the string !!")

newstr=list(string)

newlist=[]

for j in newstr:

    if j not in newlist:

        newlist.append(j)

        count=0

        for i in range(len(newstr)):

            if j==newstr[i]:

                count+=1

        print("{},{}".format(j,count))


# In[ ]:




