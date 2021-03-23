import random
import numpy as np
from cv2 import cv2
from PIL import Image
import pandas as pd
from Supermarket_class import Supermarket


class Customer:
    """Customer shops at the supermarket"""
    def __init__(self, trans_matrix, starting_state_space, image):
        self.trans_matrix = trans_matrix
        self.image = image
        self.locations =['dairy','drinks', 'fruit', 'spices']
        self.starting_state_space = starting_state_space
        self.current_location = [650, random.randint(680, 880)]
        self.target_depart = np.random.choice(self.locations, p=self.starting_state_space)
        self.target_y, self.target_x = self.calc_coordination(self.target_depart)
        self.speed = 1
        self.is_checked_out = False
        self.checkout_waiting=0
        self.money_to_spend= random.randint(1,100)
       
    def calc_coordination(self, depart):
        """get coordinates for departments in the supermarket"""

        if depart == 'drinks':
            target_y, target_x = [random.randint(135, 435), random.randint(65, 160)]
        elif depart == 'dairy':
            target_y, target_x = [random.randint(135, 435), random.randint(290, 390)]
        elif depart == 'spices':
            target_y, target_x = [random.randint(135, 435), random.randint(530, 620)]
        elif depart == 'fruit':
            target_y, target_x = [random.randint(135, 435), random.randint(750, 845)]
        elif depart == 'checkout':
            target_y, target_x = [550, random.choice([97, 247, 388, 525])]
        return target_y, target_x


    def next_target(self, depart):
        """find the next target from an aisle depends on the transition probability matrix"""
        depart_probas = self.trans_matrix.loc[depart]
        self.target_depart = np.random.choice(depart_probas.index, p=depart_probas.values)
        self.target_y, self.target_x = self.calc_coordination(self.target_depart)


    def customer_move(self):
        
        y, x = self.current_location
        # if target is checkout:
        if self.target_y == 550:
            if y == self.target_y and x == self.target_x:
                if self.checkout_waiting < 200:
                    self.checkout_waiting += 1
                elif self.checkout_waiting == 200:
                    self.image = np.zeros((35,25,3), dtype=np.uint8)
                    self.image[:,:,0:3] = 255
                    if self.is_checked_out == False:
                        customers.append(Customer(transition_matrix, initial_state_vector, customer_image))
                        self.is_checked_out=True
                                
            elif y < 470:
                self.current_location[0] += self.speed # go down
            elif y >= 470:
                if  x == self.target_x:
                    self.current_location[0] += self.speed # go down
                elif x > self.target_x:
                    self.current_location[1] -= self.speed # go left
                elif x < self.target_x:
                    self.current_location[1] += self.speed # go right
            elif y < 555:
                self.current_location[0] += self.speed # go down

        # if goal-department is reached:
        elif x == self.target_x and y == self.target_y:
            self.next_target(self.target_depart)

        # if customer is not exactly under/over target x
        elif x != self.target_x:
            if y > 450:
                if self.target_x > x:
                    self.current_location[0] -= self.speed # go up
                elif self.target_x < x:
                    self.current_location[0] -= self.speed # go up
            elif y == 450:
                if self.target_x > x:
                    self.current_location[1] += self.speed # go right
                elif self.target_x < x:
                    self.current_location[1] -= self.speed # go left
            elif y == 100:
                if self.target_x > x:
                    self.current_location[1] += self.speed  # go right
                elif self.target_x < x:
                    self.current_location[1] -= self.speed # go left
            elif y > 100:
                if abs(100 - y) < abs(450 - y):
                    self.current_location[0] -= self.speed # go up
                else:
                    self.current_location[0] += self.speed # go down

        # if customer is right under/over target x
        elif x == self.target_x:
            if y == 100:
                self.current_location[0] += self.speed  # go down
            elif y < 450:
                if y > 100:
                    if self.target_y < y:
                        self.current_location[0] -= self.speed  # go up
                    elif self.target_y > y:
                        self.current_location[0] += self.speed  # go down
            elif y == 450:
                self.current_location[0] -= self.speed  # go up
    
    

def fill_fruit(market_image):
    fruit_image = Image.open('fr.jpg')
    market = np.array(market_image)
    fruit = np.array(fruit_image)
    market[230:430,880:930]=fruit
    im = cv2.cvtColor(market, cv2.COLOR_RGB2BGR)
    return im 



if __name__ == '__main__':

    # read transition matrix and initial state vector from csv files
    transition_matrix = pd.read_csv('output/transition_matrix.csv',index_col=0,sep=',')
    initial_state = pd.read_csv('output/initial_state_vector.csv')
    initial_state_vector =initial_state["customer_no"]
    

    #number of customers that are allowed in the same time
    customers_num = 10

    customer_image = cv2.imread('customer.png')
    supermarket_image = cv2.imread('market.png')
    #print(supermarket_image.shape)
    supermarket_image = fill_fruit(supermarket_image)
    customer_image = cv2.resize(customer_image,(35,25))
    
    #creating customer-instances:
    customers = []
    for _ in range(int(customers_num)):
        customers.append(Customer(transition_matrix,initial_state_vector,customer_image))
    revenue = 0
    # creating supermarket-instance:
    supermarket = Supermarket(supermarket_image, customers,revenue)
    
    #parameters for text: to add revenue to the background
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (70, 40)
    fontScale = 0.8
    FONT_COLOR = (1, 1, 1)
    LINE_TYPE = 2

    # move the customers and keep drawing the supermarket
    while True:
        supermarket.draw(customers)
        for customer in customers:
            customer.customer_move()
            if(customer.is_checked_out == True):
                revenue += customer.money_to_spend
        cv2.rectangle(supermarket.background,(60,0),(300,50),(0,0,0),6)
        #print(f'Revenue: {revenue}')

        cv2.putText(
        supermarket.frame,
        f'Revenue: {revenue} $',
        bottomLeftCornerOfText,
        font,
        fontScale,
        FONT_COLOR,
        LINE_TYPE,
        )

        # delete customer after check out  
        customers = [customer for customer in customers if customer.is_checked_out == False]
            
        cv2.imshow('frame', supermarket.frame)

        # simulation stops if 'q' is pressed
        # control the speed of simulation by defining time(in ms) between images show
        if cv2.waitKey(7) & 0xFF == ord('q'):  
            break

    cv2.destroyAllWindows()
