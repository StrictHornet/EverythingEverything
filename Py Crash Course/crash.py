class Institution:
 def __init__(self, name, field):
  self.name = name
  self.field = field
  
 def about(self):
  print(f'{self.name} is a {self.field} intstitution.')
  
class Company(Institution):
 pass
 
 
nameOfComp = input("What is your company's name:")
fieldOfComp = input("What is your company's field:")

company = Company(nameOfComp, fieldOfComp)
company.about()