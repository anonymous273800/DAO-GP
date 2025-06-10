class MBPerformance:
    def __init__(self, epoch, r2, cost):
        self.epoch = epoch
        self.r2 = r2
        self.cost = cost

    def set_r2(self, r2):
        self.r2 = r2

    def get_r2(self):
        return self.r2

    def set_cost(self, cost):
        self.cost = cost

    def get_cost(self):
        return self.cost

    def __str__(self):
        return "epoch: "+self.epoch + " " + "r2: {:.5f}".format(self.r2) + " " + "mse: {:.5f}".format(self.cost)
