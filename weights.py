import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dist_gen import get_distance
from colorama import Fore, Back, Style
from scipy.integrate import nquad
distances, _, state_values, current_state, probs, actions, agent_position, hazard_position = get_distance.dp_net(1,1,1,1, 1)

#define device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define hyper parameters
inputs = current_state
hidden_size = 128
weights = probs
lb = 0.01 #the trushhold for lower_bound 
ub = weights.max() #the maximum value of distance 
class WeightNet(nn.Module):
    def __init__(self, inputs, hidden_size, weights):
        """[summary]
        Args:
            inputs ([state_value]): [A dictionary containing co-ordinates]:
                                {'accelerometer': array(shape[X, Y, Z]), 'velocimeter': array(shape[X, Y, Z]), 
                                'gyro': array(shape[X, Y, Z]), 'magnetometer': array(shape[X, Y, Z]), 
                                'ballangvel_rear': array(shape[X, Y, Z]), 'ballquat_rear': array([shape[X, Y, Z],
                                [0., 1., 0.],[0., 0., 1.]]), 'hazards_lidar': array(shape[number of sensors ]), 
                                'vases_lidar': array(shape[number of vases sensors])}
            hidden_size ([int]): [the size of the hidden layer]
            weights ([tensor]): [weights are the distance of robots's each senros to the closest object]
        """  
        
        super(WeightNet, self).__init__()
        """[initalise the variables]
        """       
        self.inputs = inputs
        self.weights = weights
        self.hidden_size = hidden_size
        
        #define the linear layers
        self.fc1 = nn.Linear(current_state.shape[1], hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, weights.shape[1])
        
    def forward(self, x):
        """[The forward function takes in the inputs and run it 
        through the linear linear and applies the non-linear activations]
        Args:
            x ([tensor batch_size]): [description]
        Returns:
            [tensor]: [batch_size prediction for weights]
        """        
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # to return a vlaue between [0, 1]
        x = F.sigmoid(self.fc3(x))
        #x = self.fc3(x)
        return x
    
#evaluate the true asscignments
bool_var = []    
def lit_weight_true(weights): #this wont work if we increase num_bins
    for i in weights:
        if lb <= i and i <= ub:
            bool_var.append(1)
        else:
            bool_var.append(0) 
#bool_var.append(i)
    #print("LETS CHEEEK THIS:!")
    #import pdb; pdb.set_trace()
    return bool_var
bool_var_pred = []
def lit_weight_pred(pred):
    #return torch.logical_and(pred >= ub, pred <= lb) #for more than one sensor
    for i in pred:
        if lb <= i and i <= ub:
            bool_var_pred.append(1)
        else:
            bool_var_pred.append(0)
    return bool_var_pred 

def construct_dist(pred):
    mean = np.mean(pred)
    std = np.std(pred) #this should be sigma as we only have samples 
    gauss = np.random.normal(loc=mean, scale=std, size=len(pred))
    #print("this is the full distiebution:", gauss)
    count, bins, ignored = plt.hist(gauss, len(pred), density=True)
    #sigma = std
    #mu = mean
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    #      linewidth=2, color='r')
    #plt.show() #uncomment to see the density
    mean = gauss.mean()
    var = gauss.std()
    #print("this is the  mean, var:", mean, "this is the variance:", var)

    return gauss , mean, var

# def sample_dist(gaussl):
#     #samples only 1D array for 2d Array use generator.choice
#     mu, var = np.random.randn(gaussl.mean(), gaussl.std())
#     print("this is the samples:", mu, var)
    
#     return mu, var 
def Integrand(x):
    
    sigma = var
    mu = mean
    #print("this is the  mean, var:", mu, "this is the variance:", var)
    density = (1/(sigma*np.sqrt(2*np.pi))) * \
        (np.exp((-1/2)*(((x-mu)/sigma)**2)))
    # weight_theta = (1/(self.sigma[1]*np.sqrt(2*np.pi))) * \
    #     (np.exp((-1/2)*(((y-self.mu[1])/self.sigma[1])**2)))
    return density #*weight_theta
        # return (1/(x_sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*((x-x_mu)/x_sigma)**2)*(1/(theta_sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*((x-theta_mu)/theta_sigma)**2)

def Integrate(self, mu, var):
    #self.set_mu_sigma(mu, log_var)

    ans, err = nquad(self.Integrand, [lb, ub])
    ans = float(format(ans, '.6f'))  # change .xf for x number of decimals
    return ans # [0, 1] 

        
#define model          
network =  WeightNet(inputs, hidden_size, weights)
#loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
total_steps = 1000
label = torch.tensor(weights, dtype=torch.float32) #.to(device)
for i in range(1000):

    #forward pass
    pred = network(inputs)
    true = torch.tensor(lit_weight_true(weights), dtype=torch.float32)
    lit_weight_pred(pred)
    gauss, mean, var = construct_dist(pred.detach().numpy())
    Integrand(x=1)
    label = label
    loss = criterion(pred, true)
    #import pdb; pdb.set_trace()
    #backward pass and optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1) % 10  == 0:
        print('Step [{}/{}], Actual / Predicted:[{}/{}], Loss :{:.4f}'
              .format(i+1, total_steps, bool_var[1], bool_var_pred[1], loss.item()))

#plot 

# predicted = network(inputs).detach().numpy()
# plt.plot(current_state, weights, 'ro')
# plt.plot(current_state, predicted, 'b')
# plt.show()

#test model w
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     # for images, labels in test_loader:
#     #     images = images.reshape(-1, 28*28).to(device)
#     labels = labels.to(device)
#     outputs = network(inputs)
#     _, predicted = torch.max(outputs.data, 1)
#     total += label.size(0)
#     correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test iteration: {} %'.format(100 * correct / total))  

class TheoryChecker(object):
    __symbols_map = {}

    def __init__(self, theory, literals):
        """[summary]
            theory = (0.1 <= l1 and l2 <= )
                   = (l1 (distance 1st sensor) and )
            
        """
        self._theory = theory
        self._literals = literals
        self._checked_satisfactions = False
        return

    def satisfactions(self):
        """[summary]
        This function calculates whether the constraints for each of the literals are satisfied.
        Then it generates a returns a list, the entries of which correspond to satisfaction of each literal's constraints.
        This functions needs to be called before all_satisfied and weighted_satisfaction. 
        If not then those functions call it automatically as they require the calculations performed here.
        Returns:
            List: A list of booleans corresponding to satisfaction of each literal's constraints
        """
        self._literal_satisfactions = []
        # Go through each literal and its constraints
        for literal, literal_theory in zip(self._literals, self._theory): 
            satisfied = True # Assume everything is satisfied and check for non-satiscations
            if literal_theory[0][0] is not None: # Check if there is a constraint in the first slot
                # Select the  constraint type (<=, >=, <, >, ==)
                if literal_theory[0][0] == 'le':
                    if not literal <= literal_theory[1][0]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][0] == 'ge':
                    if not literal >= literal_theory[1][0]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][0] == 'lt':
                    if not literal < literal_theory[1][0]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][0] == 'gt':
                    if not literal > literal_theory[1][0]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][0] == 'eq':
                    if not literal == literal_theory[1][0]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
            
            if literal_theory[0][1] is not None: # Same as before for the second slot.
                if literal_theory[0][1] == 'le':
                    if not literal <= literal_theory[1][1]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][1] == 'ge':
                    if not literal >= literal_theory[1][1]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][1] == 'lt':
                    if not literal < literal_theory[1][1]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][1] == 'gt':
                    if not literal > literal_theory[1][1]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue
                elif literal_theory[0][1] == 'eq':
                    if not literal == literal_theory[1][1]:
                        satisfied = False
                        self._literal_satisfactions.append(satisfied)
                        continue 
            # Everything was satisfied for this literal if this line was reached.
            self._literal_satisfactions.append(satisfied)

        self._checked_satisfactions = True # Flag up. The calculation has been performed.
        return self._literal_satisfactions
    
    def all_satisfied(self):
        """[summary]
        Checks if the constraints for all literals are satisfied. Thus, whether the whole theory is satisfied.
        Returns:
            Boolean: Whether all the constraints for all literals are satisfied.
        """
        if not self._checked_satisfactions:
            self.satisfactions()
        return all(self._literal_satisfactions)

    def weighted_satisfaction(self, weights):
        """[summary]
        Returns the weighted satisfaction sum of the literals weighted by the given parameter.
        Args:
            weights ([type]): The weight of each literal's satisfaction. Expecting an Numpy array but will also work with a python list.
        """
        if not self._checked_satisfactions:
            self.satisfactions()
        sats = np.array(self._literal_satisfactions)
        return np.sum(sats * weights * self._literals)


# if __name__ == "__main__":
#     # Assuming two literals.
#     theory = [
#             (('le', 'ge'), (0.01, weight.max())), 
#             # theory = (0.01 <= l1 >= weight.max()) and (0.01 <= l >= weight.max())

#     theory = [ 
#             # l1 >= 0.3 and l1 <= 0.6
#             (('ge', 'le'), (0.3, 0.6)), # Rules for literal 1
#             # NO RULE and l2 >= 0.3
#             ((None, 'ge'), (None, 0.3))  # Rules for literal 2
#         ]
#     # Create a theory checker with the theory you want to check and the current literal values.
#     theory_checker = TheoryChecker(theory, [0.7, 0.4]) 
#     print('All satisfied:', theory_checker.all_satisfied())
#     print('Satisfactions:', theory_checker.satisfactions())
#     print('Weighted satisfactions:', theory_checker.weighted_satisfaction([0.3, 0.7]))