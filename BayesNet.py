import numpy as np
import pandas as pd
from itertools import product

from DiscreteFactors import Factor
from Graph import Graph

def allEqualThisIndex(dict_of_arrays, **fixed_vars):
    """
    Helper function to create a boolean index vector into a tabular data structure,
    such that we return True only for rows of the table where, e.g.
    column_a=fixed_vars['column_a'] and column_b=fixed_vars['column_b'].
    
    This is a simple task, but it's not *quite* obvious
    for various obscure technical reasons.
    
    It is perhaps best explained by an example.
    
    >>> all_equal_this_index(
    ...    {'X': [1, 1, 0], Y: [1, 0, 1]},
    ...    X=1,
    ...    Y=1
    ... )
    [True, False, False]
    """
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
    return index

def estimateFactor(data, var_name, parent_names, outcomeSpace):
    """
    Calculate a dictionary probability table by ML given
    `data`, a dictionary or dataframe of observations
    `var_name`, the column of the data to be used for the conditioned variable and
    `parent_names`, a tuple of columns to be used for the parents and
    `outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
    Return a dictionary containing an estimated conditional probability table.
    """    
    var_outcomes = outcomeSpace[var_name]
    parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    f = Factor(list(parent_names)+[var_name], outcomeSpace)
    
    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        parent_index = allEqualThisIndex(data, **parent_vars)
        for var_outcome in var_outcomes:
            var_index = (np.asarray(data[var_name])==var_outcome)
            f[tuple(list(parent_combination)+[var_outcome])] = (var_index & parent_index).sum()/parent_index.sum()
            
    return f




class BayesNet():
    def __init__(self, graph, outcomeSpace=None, factor_dict=None):
        self.graph = graph
        self.outcomeSpace = dict()
        self.factors = dict()
        if outcomeSpace is not None:
            self.outcomeSpace = outcomeSpace
        if factor_dict is not None:
            self.factors = factor_dict
    
    def learnParameters(self, data):
        '''
        Iterate over each node in the graph, and use the given data
        to estimate the factor P(node|parents), then add the new factor 
        to the `self.factors` dictionary.
        '''
        graphT = self.graph.transpose()
        for node, parents in graphT.adj_list.items():
            # TODO estimate each factor and add it to the `self.factors` dictionary
            self.factors[node] = estimateFactor(data, node, parents, self.outcomeSpace)
    
    def joint(self):
        '''
        Join every factor in the network, and return the resulting factor.
        '''
        factor_list = list(self.factors.values())
        
        accumulator = factor_list[0]
        for factor in factor_list[1:]:
            accumulator = accumulator.join(factor)
        
        return accumulator
    
    def query(self, q_vars, **q_evi):
        """
        arguments 
        `q_vars`, list of variables in query head
        `q_evi`, dictionary of evidence in the form of variables names and values

        Returns a new NORMALIZED factor will all hidden variables eliminated as evidence set as in q_evi
        """     

        # first we calculate the joint distribution
        f = self.joint()
        
        # Next, we set the evidence 
        f = f.evidence(**q_evi)

        # Second, we eliminate hidden variables NOT in the query
        var_to_eliminate = list(self.graph.adj_list.keys() - q_vars - q_evi.keys())
        # print(var_to_eliminate)
        for v in var_to_eliminate:
            f = f.marginalize(v)
        
        # Finally, we normalize, then return the factor
        return f.normalize()
    
    
    
    
    
    
    
    
            
            
            
            
       