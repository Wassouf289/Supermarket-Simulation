class Supermarket:

    def __init__(self, background, customers,revenue):
        self.background = background
        self.customers = customers
        self.revenue = revenue
        
    def draw(self, customers):
        self.frame = self.background.copy()
        for customer in customers:
            y, x = customer.current_location
            self.frame[y:y+customer.image.shape[0], x:x+customer.image.shape[1]] = customer.image



        
