# Global imports
import json
import numpy as np

class InstanceDataBuilder:
    """
    Takes an instance (nominal data) of a bilevel knapsack problem with
    interdiction constraints and return the data of the Gamma-robust variant
    of the problem with uncertain objective function coefficients.
    """
    def __init__(self,
                 instance_file,
                 conservatism,
                 uncertainty=None,
                 deviations=None):
        self.instance_file = instance_file
        self.uncertainty = uncertainty
        self.deviations = deviations
        
        # The level of conservatism must take values between 0 and 1.
        if conservatism < 0 or conservatism > 1:
            raise ValueError('Level of conservatism must be between 0 and 1.')
        self.conservatism = conservatism

    def build_robustified_instance(self):
        # Read (nominal) instance data.
        instance_data = self.read_instance()

        # Add level of conservatism and deviations for robustification.
        instance_data['gamma'] = self.add_conservatism_level(instance_data)
        instance_data['deviations'] = self.add_deviations(instance_data)

        # Sort indices such that the deviations are non-increasing.
        deviations, order = self.sort_indices(instance_data)
        instance_data['deviations'] = deviations
        
        # Update instance data using new order of indices.
        keys = ['profits', 'leader weights', 'follower weights',
                'leader costs', 'follower costs']
        for key in keys:
            if key in instance_data:
                instance_data[key] = instance_data[key][order]
        
        # Construct modified profits.
        instance_data['modified profits']\
            = self.add_modified_profits(instance_data)
        return instance_data
        
    def read_instance(self):
        with open(self.instance_file, 'r') as file:
            data_from_file = file.read()
        file.close()
        
        instance_data = json.loads(data_from_file)
        keys = ['profits', 'leader weights', 'follower weights',
                'leader costs', 'follower costs']
        for key in keys:
            if key in instance_data:
                instance_data[key] = np.asarray(instance_data[key])

        if 'size' not in instance_data:
            instance_data['size'] = len(instance_data['profits'])
                
        # Check if the given data has appropriate dimensions.
        if len(instance_data['follower weights']) != instance_data['size']:
            raise ValueError('Dimensions do not match (follower weights).')
        
        if len(instance_data['leader weights']) != instance_data['size']:
            raise ValueError('Dimensions do not match (leader weights).')
        
        if len(instance_data['profits']) != instance_data['size']:
            raise ValueError('Dimensions do not match (profits).')

        # Check requirements for applicability of the methods.
        if (instance_data['follower weights'] < 0).any():
            raise ValueError('Follower weights must be non-negative.')
        
        if (instance_data['leader weights'] < 0).any():
            raise ValueError('Leader weights must be non-negative.')

        if instance_data['follower budget'] < 0:
            raise ValueError('Follower budget must be non-negative.')

        if instance_data['leader budget'] < 0:
            raise ValueError('Leader budget must be non-negative.')

        return instance_data

    def add_conservatism_level(self, instance_data):
        size = instance_data['size']
        gamma = int(np.round(self.conservatism*size))
        return gamma

    def add_deviations(self, instance_data):
        if self.deviations is not None:
            self.deviations = np.asarray(self.deviations)
            
            if (self.deviations < 0).any():
                raise ValueError('Deviations must be non-negative.')
            
            if len(self.deviations) != instance_data['size']:
                raise ValueError('Dimensions do not match (deviations).')
            
            if self.uncertainty is not None:
                raise ValueError('Either specify uncertainty or deviations.')
            return self.deviations

        if self.uncertainty is not None:
            if ((self.uncertainty < 0) or (self.uncertainty > 1)):
                raise ValueError('Uncertainty must be between 0 and 1.')
            profits = instance_data['profits']
            deviations = self.uncertainty*profits
            return deviations
        raise ValueError('Either specify uncertainty or deviations.')

    def sort_indices(self, instance_data):
        deviations = instance_data['deviations']
        order = np.argsort(-deviations)
        deviations = deviations[order]
        deviations = np.append(deviations,0)
        return deviations, order
    
    def add_modified_profits(self, instance_data):
        size = instance_data['size']
        profits = instance_data['profits']
        deviations = instance_data['deviations']
        
        modified_profits = np.zeros((size + 1, size))
        for follower in range(size + 1):
            for idx in range(follower):
                modified_profits[follower, idx] = (profits[idx]
                                                   - deviations[idx]
                                                   + deviations[follower])
            for idx in range(follower, size):
                modified_profits[follower, idx] = profits[idx]
        return modified_profits
