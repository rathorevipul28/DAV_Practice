
# coding: utf-8

# In[1]:


import numpy as np

def feature_sign_search(A, y, gamma):
    basis_matrix = np.matmul(A.T, A)
    x = np.zeros(basis_matrix.shape[0])
    desired_correlation = np.dot(A.T, y)
    theta_vector = np.zeros(basis_matrix.shape[0])
    grad = (- 2 * desired_correlation) + (2 * np.matmul(basis_matrix, x))
    active_set = set()
    zero_opt_grad = 99999999999
    nonzero_opt_grad = 0
    while not((zero_opt_grad <= gamma) and (np.allclose(nonzero_opt_grad, 0))):     
        if np.allclose(nonzero_opt_grad, 0):
            i = np.argmax(np.abs(grad) * (theta_vector == 0))
            if grad[i] > gamma:
                theta_vector[i] = -1.
                x[i] = 0
                active_set.add(i)
            elif grad[i] < -gamma:
                theta_vector[i] = 1.
                x[i] = 0
                active_set.add(i)
            if len(active_set) == 0:
                break
        else:
            continue
        index = np.array(sorted(active_set))
        constr_theta_vector = theta_vector[index]
        constr_basis = basis_matrix[np.ix_(index, index)]
        constr_correlation = desired_correlation[index]
        x_dash = np.linalg.solve(constr_basis, (constr_correlation - gamma * constr_theta_vector / 2))
        new_theta_vector = np.sign(x_dash)
        constr_old_x = x[index]
        theta_change_vector = np.where(abs(new_theta_vector - constr_theta_vector) > 1)[0]
        if len(theta_change_vector)!= 0:
            lowest_curr = x_dash
            lowest_obj = (np.dot(y.T, y) + (np.dot(x_dash, np.dot(constr_basis, x_dash))
                        - (2 * np.dot(x_dash, constr_correlation))) + (gamma * abs(x_dash).sum()))
            
            for j in theta_change_vector:
                a = x_dash[j]
                b = constr_old_x[j]
                prop = b / (b - a)
                curr = constr_old_x - (constr_old_x[j]) * (constr_old_x - x_dash)/(constr_old_x[j]-x_dash[j])
                cost = np.dot(y.T, y) + (np.dot(curr, np.dot(constr_basis, curr))
                              - 2 * np.dot(curr, constr_correlation)
                              + gamma * abs(curr).sum())
                
                if cost < lowest_obj:
                    lowest_obj = cost
                    lowest_prop = prop
                    lowest_curr = curr    
        else:
            lowest_curr = x_dash
        x[index] = lowest_curr
        zeros = index[np.abs(x[index]) < 1e-21]
        x[zeros] = 0.
        theta_vector[index] = np.sign(x[index])
        active_set.difference_update(zeros)
        grad = (- 2 * desired_correlation) + (2 * np.dot(basis_matrix, x))        
        nonzero_opt_grad = np.max(abs(grad[theta_vector != 0] + gamma * theta_vector[theta_vector != 0]))
        zero_opt_grad = np.max(abs(grad[theta_vector == 0]))
    return x


# In[2]:


# Testing with my own testset 
A = np.random.random((10,10))        #A matrix having basis vectors as it's columns
y = np.random.random(10)             # Input vector (Note that this is not a 1-d matrix)     
for gamma in [0.1, 0.5, 1, 2, 4, 6, 10, 20, 50, 100, 1000]:
    x = feature_sign_search(A, y, gamma) #Coefficient vector 
    ydash = np.dot(A,x)                  #predicted input vector
    print("For gamma = {0}:- percentage of error in predicted value of y is = {1}".format(gamma,(((np.linalg.norm(y-ydash))/(np.linalg.norm(y)))*100)))
    print("And the corresponding coefficient vector is -")
    print(x)
    

