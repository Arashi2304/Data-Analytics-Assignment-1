import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union

if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        return Z_0*(1 - np.exp(-L*X/Z_0))

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        L = Params[0]
        Z_0 = Params[w]
        y_pred = self.get_predictions(X, Z_0, w, L)
        return (Y - y_pred)**2
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)

def total_loss_fn(params, overs_left, runs, wickets_in_hand, model):
    # args = overs_left, runs, wickets_in_hand
    # overs_left = args[0]
    # runs = args[1]
    # wickets_in_hand = args[2]
    # model = args[3]
    square_loss = 0
    for i in range(len(wickets_in_hand)):
        square_loss += model.calculate_loss(params, overs_left[i], runs[i], wickets_in_hand[i])
    return square_loss

def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    df = pd.read_csv(data_path)
    return df


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data
    
    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    necessary_cols = ['Date','Match','Innings','Over','Runs','Wickets.in.Hand']
    new_df = data[necessary_cols]
    new_df.dropna()
    new_df = new_df[new_df['Innings']==1]
    print(new_df)
    return new_df
 

def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    params = []
    for i in range(10):
        params.append(model.Z0[i])
    params.append(model.L)    
    # print(params)
    wickets_in_hand = data['Wickets.in.Hand'].values
    runs = data['Runs'].values
    overs_left = 50 - data['Over']
    optimized_result = sp.optimize.minimize(total_loss_fn, params,args=(overs_left, runs, wickets_in_hand, model), method = 'trust-constr')
    optimized_result = optimized_result['x']
    for i in range(len(model.Z0)):
        model.Z0[i] = optimized_result[i]
    model.L = optimized_result[-1]      
    return model
    


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    balls = np.linspace(1,50,150)
    for i in range(10):
        y = model.get_predictions(balls, model.Z0[i],i+1,model.L)
        plt.plot(balls, y, label='hello')
    plt.xlabel('Overs')
    plt.ylabel('Average expected runs')
    plt.legend()
    plt.show()
    return None
    


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    model_params = []
    for i in range(10):
        model_params.append(model.Z0[i])
    model_params.append(model.L)
    print(model_params)
    return model_params


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    params = model.Z0
    params.append(model.L)
    wickets_in_hand = np.array(data['Wickets.in.Hand'])
    runs = np.array(data['Runs'])
    overs_left = 50 - np.array(data['Over'])
    square_loss = 0
    for i in range(len(wickets_in_hand)):
        square_loss += model.calculate_loss(params, overs_left[i], runs[i], wickets_in_hand[i])
    return square_loss

def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
