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
        self.Z0 = [10, 30, 40, 60, 90, 125, 150, 170, 190, 200]
        self.L = 10
    
    #DONE
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
        return Z_0 * (1 - np.exp(-L * X / Z_0))

    #DONE
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
        Z_0 = Params[w-1]
        L = Params[-1]
        Y_pred = self.get_predictions(X, Z_0, w, L)
        mse = np.mean((Y_pred - Y)**2)
        return mse
    
    #DONE
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    #DONE
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)

#DONE
def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    return pd.read_csv(data_path)

#DONE
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
    sub_df = data[['Date', 'Innings', 'Over', 'Wickets.in.Hand','Runs.Remaining']]
    for index, row in sub_df.iterrows():
        if '-' in row['Date']:
            date_parts = row['Date'].split('-')
            date_str = date_parts[0] 
        else:
            date_str = row['Date'] 
        sub_df.at[index, 'Date'] = date_str
    sub_df['Date'] = pd.to_datetime(sub_df['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
    sub_df = sub_df.dropna()
    data = sub_df[sub_df['Innings'] == 1]

    return data

#DONE
def loss_function(parameters, args):
    SE = 0
    remaining_runs = args[0]
    overs_left = args[1]
    wickets_in_hand = args[2]
    model = args[3]
    
    for i in range(len(wickets_in_hand)):
        SE += model.calculate_loss(parameters, overs_left[i], remaining_runs[i], wickets_in_hand[i])
    return SE

#DONE?
def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    remaining_runs = data['Runs.Remaining'].values
    overs_left = 50 - data['Over'].values
    wickets_in_hand = data['Wickets.in.Hand'].values
    parameters = []
    for i in range(len(model.Z0)):
        parameters.append(model.Z0[i])
    parameters.append(model.L)
    result = sp.optimize.minimize(loss_function, parameters, args=[remaining_runs, overs_left, wickets_in_hand, model], method='trust-constr')
    #print(result['fun'])
    result = result['x']
    for i in range(len(model.Z0)):
        model.Z0[i] = result[i]
    model.L = result[-1]
    #print(model.Z0,model.L)                
    return model

#DONE
def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    plt.figure()
    x = np.linspace(1,50,100)
    for i in range(len(model.Z0)):
        y = model.get_predictions(x, model.Z0[i], i+1, model.L)
        plt.plot(x,y,label=f'Z[{i+1}]: {model.Z0[i]}')
        plt.title('Run Production function')
    plt.xlim(0,50)
    plt.xlabel('Overs left')
    plt.ylabel('Average Runs Obtainable')
    plt.legend()
    plt.savefig(plot_path)
    plt.show()
    plt.close('all')
    
    return None

#DONE
def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    params = []
    for i in range(10):
        params.append(model.Z0[i])
    params.append(model.L)
    print(params)
    return None

#DONE
def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    remaining_runs = data['Runs.Remaining'].values
    overs_left = 50 - data['Over'].values
    wickets_in_hand = data['Wickets.in.Hand'].values
    parameters = []
    for i in range(len(model.Z0)):
        parameters.append(model.Z0[i])
    parameters.append(model.L)
    loss = loss_function(parameters, [remaining_runs, overs_left, wickets_in_hand, model])
    print(loss)
    return None


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
