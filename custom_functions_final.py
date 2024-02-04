import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import os
import time
import datetime
import pickle
import argparse
import itertools
from typing import Dict, List, Any, Tuple, Union
import doctest
import pandas as pd
import powerlaw
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy import stats
import powerlaw
from scipy.stats import combine_pvalues, norm
import seaborn as sns
from scipy import stats
import powerlaw
from scipy.stats import combine_pvalues, norm
import statsmodels.api as sm
from statsmodels.formula.api import ols
import glob
import ast
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import adfuller
import matplotlib.patches as mpatches
import random


def calculate_probability(num_list, threshold = 0.1):
    """"Simple calculation of default probability given a list of numbers and a threshold, returns the proportion of numbers above the threshold."""
    nums = len([i for i in num_list if i >= threshold])
    return nums / len(num_list)

def perform_pairwise_comparisons(data_to_plot1, data_to_plot2, thresholds):
    """
    Performs pairwise comparisons between two datasets across specified thresholds.
    
    This function iterates over two lists of data, `data_to_plot1` and `data_to_plot2`, comparing their group means at each index up to the minimum length of the two lists. For each comparison, it prints the result alongside the corresponding threshold from the `thresholds` list.
    
    Parameters:
    - data_to_plot1 (list): A list of datasets (e.g., lists, arrays) to compare. Each element represents a different group or condition.
    - data_to_plot2 (list): Another list of datasets to compare against `data_to_plot1`. Each element corresponds to the one in `data_to_plot1` by index.
    - thresholds (list): A list of thresholds or criteria corresponding to each comparison. Used for labeling output.
    
    Returns:
    - None. Results of comparisons are printed directly.
    """
    
    # Iterate over the datasets up to the length of the shortest list
    for i in range(min(len(data_to_plot1), len(data_to_plot2))):
        # Compare group means for the current pair of datasets
        result = compare_group_means(data_to_plot1[i], data_to_plot2[i])
        # Print the comparison result alongside the current threshold
        print(f"Comparison for Threshold {thresholds[i]}: \n {result}\n")


def group_files(directory: str = './', mode: str = 'sigma_interest_rates') -> dict:
    """
    Groups files in a specified directory based on a specified mode by parsing filenames.

    Parameters:
    - directory (str, optional): The directory to search for files. Defaults to the current directory.
    - mode (str, optional): The mode to group files by. Can be 'sigma_interest_rates' or 'sigma_exposure_node'. Defaults to 'sigma_interest_rates'.

    Returns:
    - dict: A dictionary with keys representing unique group identifiers based on the mode and values containing details about the files in each group.
    """
    grouped_files = {}  # Initialize an empty dictionary to store grouped files

    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Only process CSV files
            parts = filename.split('_')  # Split the filename into parts
            
            # Group files based on 'sigma_interest_rates' mode
            if mode == 'sigma_interest_rates':
                key = tuple(parts[:4] + parts[5:-1])  # Construct a key from parts of the filename
                if key not in grouped_files:
                    grouped_files[key] = {'filenames': [], 'sigma_interest_rates': []}
                grouped_files[key]['filenames'].append(filename)  # Add filename to the list
                grouped_files[key]['sigma_interest_rates'].append(parts[4])  # Add the relevant part to the list

            # Group files based on 'sigma_exposure_node' mode
            if mode == 'sigma_exposure_node':
                key = tuple(parts[:3] + parts[4:-1])  # Construct a key from parts of the filename differently
                if key not in grouped_files:
                    grouped_files[key] = {'filenames': [], 'sigma_exposure_node': []}
                grouped_files[key]['filenames'].append(filename)  # Add filename to the list
                grouped_files[key]['sigma_exposure_node'].append(parts[3])  # Add the relevant part to the list

    return grouped_files  # Return the dictionary containing grouped files   

def is_default(bankrupt_agents: int, threshold: float, system_size: int) -> bool:
    """Returns True if the number of bankrupt agents meets or exceeds the threshold proportion of the system size."""
    return bankrupt_agents >= threshold * system_size

def calculate_default_probability(bankrupt_agents_list: List[int], threshold: float, system_size: int) -> Tuple[float, int]:
    """
    Calculate the probability of default in the system and the number of defaults based on a list of bankrupt agents.

    Args:
    - bankrupt_agents_list (List[int]): A list containing the number of bankrupt agents for different scenarios or periods.
    - threshold (float): The threshold ratio of the system size, used to determine a default event.
    - system_size (int): The total size of the system, representing the total number of agents.

    Returns:
    - Tuple[float, int]: A tuple containing the default probability (as a float) and the total number of defaults (as an int).
    """
    defaults = [is_default(ba, threshold, system_size) for ba in bankrupt_agents_list]
    num_defaults = sum(defaults)
    default_probability = num_defaults / len(bankrupt_agents_list) if bankrupt_agents_list else 0
    return default_probability, num_defaults


def adjust_bankrupt_agents(num_bankrupt_agents_str: str, diff_bankrupt_agents_str: str) -> List[int]:
    """
    Adjusts the number of bankrupt agents based on the difference provided in string format.

    Args:
    - num_bankrupt_agents_str (str): A string representing the initial number of bankrupt agents.
    - diff_bankrupt_agents_str (str): A string representing the differences in the number of bankrupt agents over time.

    Returns:
    - List[int]: A list of adjusted numbers of bankrupt agents, starting with the initial number followed by adjustments.
    """
    num_bankrupt_agents = convert_string_to_list(num_bankrupt_agents_str)
    diff_bankrupt_agents = convert_string_to_list(diff_bankrupt_agents_str)
    # Assumes the last element of diff_bankrupt_agents is not used for adjustments.
    return [num_bankrupt_agents[0]] + diff_bankrupt_agents[:-1]

def analyze_time_series(ts1: np.ndarray, ts2: np.ndarray) -> pd.DataFrame:
    """
    Analyzes two time series by computing their correlation, testing for stationarity,
    fitting VAR and VECM models, and conducting cointegration tests.

    Parameters:
    - ts1: np.ndarray. The first time series.
    - ts2: np.ndarray. The second time series.

    Returns:
    - pd.DataFrame. A DataFrame containing the analysis results including correlation,
      ADF test statistics and p-values for each time series, VAR and VECM model diagnostics,
      and cointegration test results.
    """
    
    # Compute Pearson correlation coefficient between the two time series
    correlation = np.corrcoef(ts1, ts2)[0, 1]
    
    # Perform Augmented Dickey-Fuller test to check for stationarity in each time series
    adf_ts1 = adfuller(ts1)
    adf_ts2 = adfuller(ts2)
    
    # Combine the two time series for VAR and VECM model analysis
    combined_ts = np.column_stack((ts1, ts2))
    
    # Fit a VAR model based on the Bayesian Information Criterion (BIC) to find optimal lag
    var_model = VAR(combined_ts)
    var_model_fitted = var_model.fit(ic='bic')
    var_bic = var_model_fitted.bic  # BIC of the fitted VAR model
    var_lag_order = var_model_fitted.k_ar  # Optimal lag order
    
    # Print VAR model summary
    print(var_model_fitted.summary())
    
    # Perform Johansen's cointegration test to check for cointegration between the two time series
    johansen_test = coint_johansen(combined_ts, det_order=-1, k_ar_diff=var_lag_order-1)
    coint_rank = np.sum(johansen_test.lr1 > johansen_test.cvt[:, 1])  # Number of cointegrating relations
    
    # Initialize dictionary to store VECM results if cointegration is found
    vecm_results = {}
    if coint_rank > 0:
        # Fit a VECM model if cointegration is present
        vecm_model = VECM(combined_ts, k_ar_diff=var_lag_order-1, coint_rank=coint_rank, deterministic='ci')
        vecm_model_fitted = vecm_model.fit()
        
        # Extract and store coefficients from the fitted VECM model
        vecm_results = {
            'cointegrating_eq_coefficients': vecm_model_fitted.beta,
            'adjustment_coefficients': vecm_model_fitted.alpha,
        }
    
    # Perform diagnostics on the VAR model residuals
    var_results = {
        'var_residuals_test': var_model_fitted.test_whiteness().summary().data[1][2],  # Test for autocorrelation
    }
    
    # Combine all results into a single dictionary
    results = {
        'correlation': correlation,
        'adf_statistic_ts1': adf_ts1[0], 'adf_pvalue_ts1': adf_ts1[1],
        'adf_statistic_ts2': adf_ts2[0], 'adf_pvalue_ts2': adf_ts2[1],
        'var_bic': var_bic, 'var_lag_order': var_lag_order, 'coint_rank': coint_rank,
        **vecm_results,  # Include VECM results if any
        **var_results  # Include VAR model diagnostics
    }
    
    # Print combined results
    print(results)
    
    # Convert the results dictionary into a DataFrame for easier analysis and visualization
    results_df = pd.DataFrame([results])
    
    # Print summaries based on the presence of cointegration and stationarity tests
    if coint_rank > 0 and (adf_ts1[1] > 0.05) and (adf_ts2[1] > 0.05):
        print("\nVECM Model Summary:")
        print(vecm_model_fitted.summary())

    if coint_rank == 0:
        print("\nVAR Model Summary:")
        print(var_model_fitted.summary())
    
    return results_df

def calculate_first_order_differences(array):
    """
    Calculate the first order differences of a numpy array.

    Parameters:
    array (np.array): The input array.

    Returns:
    np.array: The first order differences of the input array.
    """
    return np.diff(array)


def brownian_motion(num_steps, delta_t, sigma):
    """
    Generate a Brownian motion path.

    Parameters:
    num_steps (int): Number of steps in the Brownian motion.
    delta_t (float): Time increment.
    sigma (float): Standard deviation of the increments (sqrt of variance).

    Returns:
    np.array: A numpy array representing the Brownian motion path.
    """
    # Generate random increments from a normal distribution
    increments = np.random.normal(0, sigma * np.sqrt(delta_t), num_steps-1)

    # The start point is typically zero
    start_point = 0

    # Compute the Brownian motion path
    return np.cumsum(np.insert(increments, 0, start_point))

def simulate_brownian_motion_one_step(exposures, delta_t, sigma):
    """
    Simulate Brownian motion for each agent's exposure for one step.

    Parameters:
    exposures (np.array): Initial exposures of the agents.
    delta_t (float): Time increment.
    sigma (float): Standard deviation of the increment.

    Returns:
    np.array: Updated exposures after one step of Brownian motion simulation.
    """
    # Generate random increments from a normal distribution for one step
    increments = np.random.normal(0, sigma * np.sqrt(delta_t), len(exposures))

    # Update exposures with the increments
    updated_exposures = exposures + increments
    return updated_exposures


def generate_exposures(N, mu=0, sigma=1):
    """
    Generate a random set of exposures for N agents.

    Parameters:
    N (int): Number of agents.

    Returns:
    np.array: A numpy array representing the exposures of the agents.
    """
    return np.random.normal(0, 1, N)


def create_directional_graph(N_Nodes, edges=None):
    """
    Creates a directed graph using NetworkX and initializes 'exposure' attribute for each node.

    Parameters:
    N_Nodes (int): The number of nodes in the graph.
    edges (list of tuples, optional): A list of edges to add to the graph.

    Returns:
    nx.DiGraph: A NetworkX directed graph with initialized 'exposure' attribute for each node.
    """
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(N_Nodes))

    # Initialize 'exposure' for each node and 'connected_this_timestep' to False
    exposures = generate_exposures(N_Nodes)
    for node, exposure in zip(G.nodes(), exposures):
        G.nodes[node]['exposure'] = exposure
        G.nodes[node]['connected_this_timestep'] = False


    # Add edges if provided
    if edges is not None:
        for edge in edges:
            G.add_edge(*edge)  # Unpack the tuple for adding an edge

    return G


def logistic_threshold_probability(x, threshold):
    """
    Compute the logistic threshold probability with fixed 'a' and 'b' values.

    Parameters:
    x (float): The value at which to compute the probability.
    threshold (float): The threshold value for scaling.

    Returns:
    float: The logistic threshold probability.
    """
    a = 1
    b = 5
    scaled_x = x * (10 / threshold)
    return 1 - 1 / (1 + np.exp(-a * (scaled_x - b)))


def get_statistics_vary_threshold(pattern: str = './200_2000_*.csv') -> Dict[float, List[float]]:
    """
    Analyze CSV files matching a specific pattern to calculate the average of the absolute differences
    of bankrupt agents over time, grouped by different threshold values.

    The function expects CSV files with a specific naming convention that includes a threshold value. 
    It calculates the average of a list contained in the 'Abs Difference of Bankrupt Agents Over Time' column 
    for each file, then groups these averages by their corresponding threshold values.

    Args:
        pattern (str): A glob pattern to match filenames. Default is './200_2000_*.csv'.

    Returns:
        Dict[float, List[float]]: A dictionary where keys are threshold values (float) and values are lists of 
        averages (float) of absolute differences for each run associated with that threshold.
    """
    abs_diff_avg_per_run_per_threshold: Dict[float, List[float]] = {}

    for filepath in glob.glob(pattern):
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath).replace('.csv', '')
        parts = filename.split('_')
        threshold_v = float(parts[5])
        abs_diff_avg = calculate_average_list(df['Abs Difference of Bankrupt Agents Over Time'])
        if threshold_v not in abs_diff_avg_per_run_per_threshold:
            abs_diff_avg_per_run_per_threshold[threshold_v] = []
        abs_diff_avg_per_run_per_threshold[threshold_v].append(abs_diff_avg)

    return abs_diff_avg_per_run_per_threshold

def sum_absolute_edge_weights(graph):
    """
    Sums the absolute values of the edge weights in a NetworkX graph.

    :param graph: A NetworkX graph with weighted edges.
    :return: The sum of the absolute values of the edge weights.
    """
    # Initialize the sum to zero
    total_weight = 0

    # Iterate over all edges in the graph
    for (u, v, weight) in graph.edges(data='weight'):
        # Sum the absolute values of the edge weights
        total_weight += abs(weight)

    return total_weight

def simulate_brownian_motion_one_step(exposures, delta_t, sigma):
    """
    Simulate Brownian motion for each agent's exposure for one step.

    Parameters:
    exposures (np.array): Initial exposures of the agents.
    delta_t (float): Time increment.
    sigma (float): Standard deviation of the increment.

    Returns:
    np.array: Updated exposures after one step of Brownian motion simulation.
    """
    # Generate random increments from a normal distribution for one step
    increments = np.random.normal(0, sigma * np.sqrt(delta_t), len(exposures))

    # Update exposures with the increments
    updated_exposures = exposures + increments
    return updated_exposures

def form_links_and_update_exposures(G: nx.DiGraph, linking_threshold: float, link_formation_mode = 'random link logic', max_one_connection_per_node=False, swap_exposure_threshold=0, time_to_maturity=0, link_threshold_mode = 'hard cutoff') -> nx.DiGraph:
    """
    This function forms links between nodes in a directed graph based on the nodes' exposure values and a specified linking threshold.
    It also updates the exposure values of these nodes according to the linking mode.

    In 'devide exposure equally' mode, the sum of the exposures of both nodes involved in a link is evenly distributed to each node.
    This means that after linking, each node will have an exposure value equal to the average of their previous exposures.
    The weight of the link created is based on the change in exposure of the nodes as a result of this process.

    In 'devide exposure singly' mode, one node's exposure is set to zero and the other node receives the entire sum of both exposures.
    This mode is not detailed in the docstring as it is less frequently used.

    Parameters:
    G (nx.DiGraph): The graph to which the nodes belong. Each node should have an 'exposure' and a 'connected_this_timestep' attribute.
    linking_threshold (float): The threshold below which the absolute sum of exposures will trigger a link formation.
    mode (str): The mode of exposure division. Defaults to 'devide exposure equally'.

    Returns:
    nx.DiGraph: The updated graph with new links and updated exposures.
    """

    if link_formation_mode == 'random link logic':
        if max_one_connection_per_node:
            # List of nodes with positive and negative exposure respectively
            positive_exposure_nodes = [node for node in G.nodes if G.nodes[node]['exposure'] > 0 and not G.nodes[node]['connected_this_timestep']]
            negative_exposure_nodes = [node for node in G.nodes if G.nodes[node]['exposure'] < 0 and not G.nodes[node]['connected_this_timestep']]

            # Randomly pair nodes from each list
            random.shuffle(positive_exposure_nodes)
            random.shuffle(negative_exposure_nodes)

            for positive_node, negative_node in zip(positive_exposure_nodes, negative_exposure_nodes):
                # Create swap only if both nodes have not connected this timestep
                if not G.nodes[positive_node]['connected_this_timestep'] and not G.nodes[negative_node]['connected_this_timestep']:
                    # Calculate the hedge value, which is the minimum of the absolute exposures or the swap value
                    hedge_value = min(abs(G.nodes[positive_node]['exposure']), abs(G.nodes[negative_node]['exposure']), swap_exposure_threshold)
                    
                    # Create the swap
                    G.add_edge(positive_node, negative_node, weight=hedge_value, time_to_maturity=time_to_maturity)
                    G.add_edge(negative_node, positive_node, weight=-hedge_value, time_to_maturity=time_to_maturity)
                    
                    # Update exposures
                    G.nodes[positive_node]['exposure'] -= hedge_value
                    G.nodes[negative_node]['exposure'] += hedge_value
                    
                    # Set connection flag if only one connection per timestep is allowed
                    if max_one_connection_per_node:
                        G.nodes[positive_node]['connected_this_timestep'] = True
                        G.nodes[negative_node]['connected_this_timestep'] = True

            # Reset the 'connected_this_timestep' attribute for the next timestep
            if max_one_connection_per_node:
                for node in G.nodes:
                    G.nodes[node]['connected_this_timestep'] = False
                    
            return G
        else:
            # List of nodes with positive and negative exposure respectively
            positive_exposure_nodes = [node for node in G.nodes if G.nodes[node]['exposure'] > 0]
            negative_exposure_nodes = [node for node in G.nodes if G.nodes[node]['exposure'] < 0]

            # Randomly pair nodes from each list
            random.shuffle(positive_exposure_nodes)
            random.shuffle(negative_exposure_nodes)

            for positive_node, negative_node in zip(positive_exposure_nodes, negative_exposure_nodes):
                # Calculate the hedge value, which is the minimum of the absolute exposures or the swap value
                hedge_value = min(abs(G.nodes[positive_node]['exposure']), abs(G.nodes[negative_node]['exposure']), swap_exposure_threshold)
                
                # Create the swap
                G.add_edge(positive_node, negative_node, weight=hedge_value, time_to_maturity=time_to_maturity)
                G.add_edge(negative_node, positive_node, weight=-hedge_value, time_to_maturity=time_to_maturity)
                
                # Update exposures
                G.nodes[positive_node]['exposure'] -= hedge_value
                G.nodes[negative_node]['exposure'] += hedge_value

            return G

    else:
        
        if linking_threshold == 0:
            linking_threshold = sys.float_info.epsilon


        for i in G.nodes:
            closest_sum = np.inf
            closest_node = None


            if max_one_connection_per_node:

                if not G.nodes[i]['connected_this_timestep']:
                    for j in G.nodes:
                        if not G.nodes[j]['connected_this_timestep'] and i != j and G.nodes[i]['exposure'] * G.nodes[j]['exposure'] < 0:
                            sum_of_exposures = G.nodes[i]['exposure'] + G.nodes[j]['exposure']
                            
                            if link_threshold_mode == 'hard cutoff':
                                if (np.abs(sum_of_exposures) < np.abs(closest_sum)) and (np.abs(sum_of_exposures) < linking_threshold):
                                    closest_sum = sum_of_exposures
                                    closest_node = j
                            if link_threshold_mode == 'logistic':
                                if (np.abs(sum_of_exposures) < np.abs(closest_sum)) and (logistic_threshold_probability(np.abs(sum_of_exposures), linking_threshold) > np.random.random()):
                                    closest_sum = sum_of_exposures
                                    closest_node = j

            else:
                    for j in G.nodes:
                        if i != j and G.nodes[i]['exposure'] * G.nodes[j]['exposure'] < 0:
                            sum_of_exposures = G.nodes[i]['exposure'] + G.nodes[j]['exposure']

                            if (np.abs(sum_of_exposures) < np.abs(closest_sum)) and ((np.abs(sum_of_exposures) < linking_threshold)):
                                closest_sum = sum_of_exposures
                                closest_node = j
        

            if closest_node and (np.abs(G.nodes[i]['exposure']) > swap_exposure_threshold):

                # deviding the exposure singly means one node gets 0 exposure and the other node gets the remaining exposure
                if link_formation_mode == 'divide exposure singly':

                    # Calculate hedge value based on the smaller absolute exposure of the two nodes
                    hedge_value = min(abs(G.nodes[i]['exposure']), abs(G.nodes[closest_node]['exposure']))

                    # Determine the sign of the weight for each edge based on the exposure of the originating node
                    weight_i_to_closest_node = np.sign(G.nodes[i]['exposure']) * hedge_value
                    weight_closest_node_to_i = np.sign(G.nodes[closest_node]['exposure']) * hedge_value

                    # Create edges with weights having signs corresponding to the exposure of the originating node
                    G.add_edge(i, closest_node, weight=weight_i_to_closest_node, time_to_maturity=time_to_maturity)
                    G.add_edge(closest_node, i, weight=weight_closest_node_to_i, time_to_maturity=time_to_maturity)
                    G.nodes[i]['exposure'] -= weight_i_to_closest_node
                    G.nodes[closest_node]['exposure'] -= weight_closest_node_to_i
                    
                    
                    # Setting the connection flag for both nodes
                    G.nodes[i]['connected_this_timestep'] = True
                    G.nodes[closest_node]['connected_this_timestep'] = True


                # deviding the exposure equally means that the sum of the exposure of both nodes is evenly distributed to each node.
                if link_formation_mode == 'divide exposure equally':
                    # Calculate the average exposure to evenly divide between the two nodes
                    average_exposure = (G.nodes[closest_node]['exposure'] + G.nodes[i]['exposure']) / 2
                    # Determine the weight of the link based on the change in exposure
                    weight = G.nodes[i]['exposure'] - average_exposure
                    # Update exposures and add weighted edges
                    G.nodes[i]['exposure'] = average_exposure
                    G.nodes[closest_node]['exposure'] = average_exposure
                    G.add_edge(i, closest_node, weight=weight, time_to_maturity=time_to_maturity)
                    G.add_edge(closest_node, i, weight=-weight, time_to_maturity=time_to_maturity)
                    
                    # Setting the connection flag for both nodes
                    G.nodes[i]['connected_this_timestep'] = True
                    G.nodes[closest_node]['connected_this_timestep'] = True

                    
                    
        # Resetting the flag for the next timestep
                    
        if max_one_connection_per_node:
            for node in G.nodes:
                G.nodes[node]['connected_this_timestep'] = False


        return G


def check_bankruptcy_and_update_network(G, threshold_v, delta_price, create_new_node_mode=True, bankruptcy_mode='exposure'):
    """
    Analyze a network of nodes (represented by a graph) to identify and process bankrupt nodes based on a volatility threshold.
    This function updates the exposures of nodes in the network, identifies bankrupt nodes, redistributes their exposures to neighbors,
    and optionally adds new nodes to the network.

    :param G: The graph representing the network of nodes.
    :param threshold_v: The threshold for volatility, above which a node is considered bankrupt.
    :param delta_price: The change in price, used to calculate volatility.
    :param create_new_node_mode: Flag to determine whether to create a new node for each bankrupt node removed.
    :return: A tuple containing the updated graph and the number of bankruptcies identified.
    """
    bankrupt_nodes = set()
    num_bankruptcies = 0
    edges_to_remove = []
    edges_to_decrement = []

    # Process edges with zero time to maturity
    for u, v, attr in G.edges(data=True):
        time_to_maturity = attr.get('time_to_maturity', 1)  # Assuming default time_to_maturity is 1 if not present

        if time_to_maturity == 0:
            exposure_u = G.nodes[u]['exposure']
            weight = attr['weight']

            # Adjust exposures considering the directionality
            G.nodes[u]['exposure'] = exposure_u + weight

            # Mark the edge for removal
            edges_to_remove.append((u, v))
        else:
            # Mark the edge for decrementing time to maturity
            edges_to_decrement.append((u, v, time_to_maturity))

    # Remove marked edges
    for u, v in edges_to_remove:
        G.remove_edge(u, v)
        

    # Decrement time to maturity for marked edges
    for u, v, time_to_maturity in edges_to_decrement:
        G.edges[u, v]['time_to_maturity'] = max(0, time_to_maturity - 1)



    # Update exposures based on volatility and identify bankrupt nodes
    for node in G.nodes():
        exposure = G.nodes[node].get('exposure', 1)  # Get exposure from node, defaulting to 1 if not present
        if bankruptcy_mode == 'exposure':
            volatility = exposure
        elif bankruptcy_mode == 'intrest_rate':
            volatility = exposure * delta_price
        if abs(volatility) > threshold_v:
            bankrupt_nodes.add(node)
            num_bankruptcies += 1

    # Process bankrupt nodes
    for node in bankrupt_nodes:
        if node in G:
            # Redistribute exposure to neighbors
            for neighbor in G.neighbors(node):
                if neighbor not in bankrupt_nodes:
                    neighbor_exposure = G.nodes[neighbor].get('exposure', 0)
                    G.nodes[neighbor]['exposure'] = neighbor_exposure + G.edges[neighbor, node]['weight']

            # Remove bankrupt node
            G.remove_node(node)

            if create_new_node_mode:
                # Create new node
                new_node_id = max(G.nodes())+1 if G.nodes() else 0
                G.add_node(new_node_id, exposure=0, connected_this_timestep=False)


    return G, num_bankruptcies



def financial_network_simulator(N_agents, time_steps, delta_t, sigma_exposure_node, sigma_intrestrate, threshold_v, linking_threshold, swap_exposure_threshold=0.5, print_timestep=True, time_to_maturity=0, link_threshold_mode = 'hard cutoff', link_formation_mode = 'random link logic', bankruptcy_mode = 'exposure', create_new_node_mode = True):
    """
    Simulates a financial network over a specified number of time steps. 

    The simulation includes dynamic changes in agent exposures and inter-agent connections, 
    influenced by a stochastic process (Brownian motion). The network evolves through
    the formation of new links and the possibility of bankruptcy among agents.

    Parameters:
    - N_agents (int): Number of agents in the network.
    - num_steps (int): Number of time steps for the simulation.
    - delta_t (float): Time step size for Brownian motion.
    - sigma_exposure_node (float): Standard deviation for exposure changes of each agent.
    - sigma_intrestrate (float): Standard deviation for interest rate changes in Brownian motion.
    - threshold_v (float): Threshold value for bankruptcy determination.
    - linking_threshold (float): Threshold for forming new links between agents.
    - print_timestep (bool, optional): Flag to print the current time step (default is True).

    Returns:
    tuple: A tuple containing:
        - The final graph of the network.
        - Array of summed absolute exposures over time.
        - Array of the cumulative number of bankrupt agents over time.
        - Array of simulated prices (interest rates).
        - Array of the number of links over time.
        - Array of total absolute exposure in edge weights over time.
        - Array of node population over time.
    """
    # Initialize graph with nodes having exposure attributes
    graph = create_directional_graph(N_agents)



    # Initialize 'connected_this_timestep' flag to false for each node
    for i in range(N_agents):
        graph.nodes[i]['connected_this_step'] = False
    


    # Simulate interest rate as Brownian motion
    simulated_prices = brownian_motion(time_steps, delta_t, sigma_intrestrate)

    # Calculate price movement difference for each step
    delta_price_array = calculate_first_order_differences(simulated_prices)

    # Arrays to track metrics over time
    num_bankrupt_agents_total = 0
    num_bankrupt_agents_over_time = np.zeros(time_steps)
    links_over_time = np.zeros(time_steps)
    abs_exposures_over_time_summed = np.zeros(time_steps)
    total_abs_exposure_in_edge_weights = np.zeros(time_steps)
    node_population_over_time = np.zeros(time_steps)

    # Simulate over time
    for step in range(time_steps):

        if print_timestep:
            print('timestep', step)

        # Update exposures based on Brownian motion
        for i in graph.nodes():
            graph.nodes[i]['exposure'] += np.random.normal(0, sigma_exposure_node * delta_t)

        # Form links and update exposures based on the current state
        # graph = form_links_and_update_exposures(graph, linking_threshold, swap_exposure_threshold)
        graph = form_links_and_update_exposures(graph, linking_threshold, time_to_maturity=time_to_maturity, swap_exposure_threshold=swap_exposure_threshold, link_threshold_mode = link_threshold_mode, link_formation_mode = link_formation_mode,)

        # Check for bankruptcy and update the network
        graph, bankruptcies_this_step = check_bankruptcy_and_update_network(G = graph, threshold_v = threshold_v, delta_price = delta_price_array[step-1], create_new_node_mode = create_new_node_mode, bankruptcy_mode = bankruptcy_mode)

        # Add the number of bankruptcies this step to the total
        num_bankrupt_agents_total += bankruptcies_this_step

        # Update the number of bankruptcies over time
        num_bankrupt_agents_over_time[step] = num_bankrupt_agents_total

        # Update the number of links over time
        links_over_time[step] = graph.number_of_edges()

        # Update the sum of absolute exposures over time
        abs_exposures_over_time_summed[step] = np.sum(np.abs(np.array(list(nx.get_node_attributes(graph, 'exposure').values()))))

        # Update exposures over time
        abs_exposure_in_edge_weights = sum_absolute_edge_weights(graph)

        
        # Update the sum of absolute exposure in edge weights over time
        total_abs_exposure_in_edge_weights[step] = abs_exposure_in_edge_weights

        # Update node population over time
        node_population_over_time[step] = len(graph.nodes())

    return graph, abs_exposures_over_time_summed, num_bankrupt_agents_over_time, simulated_prices, links_over_time, total_abs_exposure_in_edge_weights, node_population_over_time

        


def multi_parameter_financial_network_simulator(runs, N_agents_list, num_steps_list, delta_t_list, sigma_exposure_node_list, sigma_intrestrate_list, threshold_v_list, linking_threshold_list, swap_exposure_threshold_list, time_to_maturity_list, link_threshold_mode_list, link_formation_mode, bankruptcy_mode, create_new_node_mode):
    """
    Runs multiple simulations of a financial network simulator for all combinations of given parameter lists. 
    Each simulation is run a specified number of times ('runs') for each combination of parameters.

    Parameters:
    runs (int): The number of times to run the simulation for each combination of parameters.
    N_agents_list (list of int): List of numbers representing the number of agents in the simulation.
    num_steps_list (list of int): List of numbers representing the number of steps in each simulation.
    delta_t_list (list of float): List of delta time values for the simulation.
    sigma_list (list of float): List of sigma values for the simulation.
    threshold_v_list (list of float): List of threshold values for bankruptcy.
    linking_threshold_list (list of float): List of threshold values for linking agents in the network.

    Returns:
    dict: A dictionary where each key is a tuple representing a combination of parameters, and each value 
    is a list of dictionaries. Each dictionary in the list contains the results of a single run of the 
    simulation for that parameter combination. The result dictionary keys are 'graph', 'exposures_over_time',
    'num_bankrupt_agents_over_time', 'simulated_prices', and 'links_over_time'.

    Example:
    To run simulations for combinations of [100, 150] agents and [50, 60] steps, each for 3 runs:
    results = multi_parameter_financial_network_simulator(3, [100, 150], [50, 60], [0.1], [0.05], [1.0], [0.5])
    
    The structure of the returned dictionary for one combination might look like this:
    {
        (100, 50, 0.1, 0.05, 1.0, 0.5): [
            {
                'graph': <networkx.DiGraph object>,
                'exposures_over_time': numpy.ndarray,
                'num_bankrupt_agents_over_time': numpy.ndarray,
                'simulated_prices': numpy.ndarray,
                'links_over_time': numpy.ndarray
            },
            # ... More runs for the same combination ...
        ],
        # ... More parameter combinations ...
    }
    """
    # Dictionary to store results
    results_dict = {}

    # Names of the parameters
    param_names = ["N_agents", "num_steps", "delta_t", "sigma_exposure_node", "sigma_intrestrate", "threshold_v", "linking_threshold", "swap_exposure_threshold", "time_to_maturity", "link_threshold_mode"]

    # Generate all combinations of parameters
    param_combinations = itertools.product(N_agents_list, num_steps_list, delta_t_list, sigma_exposure_node_list, sigma_intrestrate_list, threshold_v_list, linking_threshold_list, swap_exposure_threshold_list, time_to_maturity_list, link_threshold_mode_list)

    param_combinations_list = list(param_combinations)
    total_iterations = len(param_combinations_list) * runs

    # Initialize counter for progress bar
    iteration = 1
    # record start time
    start_time = time.time()

    for combination in param_combinations_list:
        # Unpack the combination tuple
        (N_agents, num_steps, delta_t, sigma_exposure_node, sigma_intrestrate, threshold_v, linking_threshold, swap_exposure_threshold, time_to_maturity, link_threshold_mode) = combination
        named_combination = dict(zip(param_names, combination)) # Pair each value with its parameter name
        combination_results = []

        for _ in range(runs):
            result = financial_network_simulator(N_agents, num_steps, delta_t, sigma_exposure_node, sigma_intrestrate, threshold_v, linking_threshold, swap_exposure_threshold, print_timestep=False, time_to_maturity=time_to_maturity, link_threshold_mode=link_threshold_mode, link_formation_mode = link_formation_mode, bankruptcy_mode = bankruptcy_mode, create_new_node_mode = create_new_node_mode)
            
            graph, abs_exposures_over_time_summed, num_bankrupt_agents_over_time, simulated_prices, links_over_time, total_abs_exposure_in_edge_weights, node_population_over_time = result

            run_result = {
                'graph': graph,
                'abs_exposures_over_time_summed': abs_exposures_over_time_summed,
                'num_bankrupt_agents_over_time': num_bankrupt_agents_over_time,

                'simulated_prices': simulated_prices,
                'links_over_time': links_over_time,
                'total_abs_exposure_in_edge_weights': total_abs_exposure_in_edge_weights,
                'node_population_over_time': node_population_over_time
            }
            combination_results.append(run_result)

            # Increment counter for progress bar
            iteration += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_iteration = elapsed_time / iteration
            estimated_total_time = avg_time_per_iteration * total_iterations
            time_remaining = estimated_total_time - elapsed_time

            # Format remaining time as hh:mm:ss
            remaining_time_formatted = str(datetime.timedelta(seconds=int(time_remaining)))

            # Print progress with timestamp and estimated completion time
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Progress: {iteration}/{total_iterations} ({iteration/total_iterations*100:.2f}%) - Estimated Time Remaining: {remaining_time_formatted}")

        # Save results of this run to CSV
        save_results_to_csv(combination_results, combination, link_formation_mode, bankruptcy_mode)

        # Store results for this combination
        results_dict[combination] = combination_results


    # Save to CSV
    save_all_results_to_csv(results_dict, link_formation_mode, bankruptcy_mode)

    return results_dict




# Fit the data to a power-law distribution
def fit_power_law(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a given dataset to a power-law distribution, plot the Probability Density Function (PDF) 
    for both the empirical data and the power-law fit, and compare it with an exponential distribution.

    This function uses the powerlaw package to fit the data and matplotlib for plotting. It also
    calculates the goodness of the fit by comparing the power-law distribution with an exponential
    distribution using a loglikelihood ratio test and reports the statistical significance.

    Parameters:
    data (np.ndarray): A one-dimensional Numpy array of the data to be fitted.

    Returns:
    Tuple[float, float, float]: A tuple containing the power-law exponent (alpha), the loglikelihood 
    ratio (R), and the p-value (p) of the comparison between the power-law and exponential distributions. 
    If R > 0 and p < 0.05, the data is better explained by a power-law distribution.
    """
    # Fit the data to a power-law distribution
    results = powerlaw.Fit(data)
    
    # Extract the power-law exponent
    alpha = results.power_law.alpha

    # Plot the PDF and compare it to the power-law.
    plt.figure(figsize=(12, 8))
    results.power_law.plot_pdf(color='b', linestyle='--', label='Power-law fit')
    results.plot_pdf(color='b', label='Empirical Data')
    plt.xlabel('Data')
    plt.ylabel('Probability Density Function (PDF)')
    plt.title('PDF and Power-law Fit of the Data')
    plt.legend()
    plt.show()

    # Calculate the goodness of fit
    R, p = results.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    
    # Output the loglikelihood ratio and p-value
    print(f"Power-law exponent (alpha): {alpha}")
    print(f"Loglikelihood ratio between power law and exponential distribution: {R}")
    print(f"Statistical significance of the fit (p-value): {p}")

    # Conclusion based on the loglikelihood ratio, p-value, and alpha
    if R > 0 and p < 0.05:
        print("The data follows a power-law distribution better than an exponential distribution.")
    else:
        print("There is not sufficient evidence to conclude that the data follows a power-law distribution better than an exponential distribution.")

    return alpha, R, p

def draw_graph_with_edge_weights(G, pos=None, node_size=400, node_color='skyblue', font_size=10,
                                 font_weight='bold', arrowstyle='-|>', arrowsize=30, width=2,
                                 edge_precision=3, offset=0.1, label_offset_x=0.02, label_offset_y=0.05):
    """
    Draw a directed graph with edge weights rounded to a specified precision and display weights for bidirectional edges.
    The weight labels will be placed above or below the edge based on the direction.
    Parameters:
    G (nx.DiGraph): The directed graph to draw.
    pos (dict): Position coordinates for nodes for specific layout.
    node_size (int): Size of nodes.
    node_color (str): Color of nodes.
    font_size (int): Font size for node labels.
    font_weight (str): Font weight for node labels.
    arrowstyle (str): Style of the arrows.
    arrowsize (int): Size of the arrows.
    width (int): Width of edges.
    edge_precision (int): Decimal precision for edge weights.
    offset (float): Offset for edge labels to prevent overlap.
    label_offset_x (float): Horizontal offset for labels in bidirectional edges.
    label_offset_y (float): Vertical offset for labels in bidirectional edges.
    Returns:
    None: Draws the graph with matplotlib.
    """

    if pos is None:
        pos = nx.spring_layout(G)  # positions for all nodes

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight=font_weight)

    # Draw edges with arrows
    for edge in G.edges():
        source, target = edge
        rad = 0.1
        arrow = FancyArrowPatch(pos[source], pos[target], arrowstyle=arrowstyle,
                                connectionstyle=f'arc3,rad={rad}', mutation_scale=arrowsize,
                                lw=width, color='black')
        ax.add_patch(arrow)

    # Draw edge weights with precision and apply offsets
    for edge in G.edges():
        source, target = edge
        weight = G[source][target]['weight']
        x_offset = (pos[target][0] - pos[source][0]) * offset
        y_offset = (pos[target][1] - pos[source][1]) * offset
        text_pos = ((pos[source][0] + pos[target][0]) / 2 + x_offset,
                    (pos[source][1] + pos[target][1]) / 2 + y_offset)

        # Adjust label position based on the direction of the edge
        if source < target:
            text_pos = (text_pos[0], text_pos[1] + label_offset_y)
        else:
            text_pos = (text_pos[0], text_pos[1] - label_offset_y)

        ax.text(*text_pos, s=f'{weight:.{edge_precision}f}', horizontalalignment='center',
                verticalalignment='center', fontsize=font_size, fontweight=font_weight)

    plt.axis('off')
    plt.show()


    
def plot_financial_network_results(num_bankrupt_agents_over_time, node_population_over_time, links_over_time, total_abs_exposure_in_edge_weights, exposures_over_time, simulated_prices):

    # Create an array of time steps, can choose any of the arrays as they all have the same length
    time_steps = np.arange(len(num_bankrupt_agents_over_time))

     # Calculate network density for each time step
    network_density_over_time = [links_over_time[t] / (node_population_over_time[0] * (node_population_over_time[0] - 1)) for t in range(len(time_steps))]

    # Calculate the difference in the number of bankrupt agents
    diff_bankrupt_agents = np.diff(num_bankrupt_agents_over_time)

    # Create a large figure to hold all subplots
    plt.figure(figsize=(12, 24))

    # Subplot 1: Number of Bankrupt Agents Over Time
    plt.subplot(5, 2, 1)
    plt.plot(time_steps, num_bankrupt_agents_over_time, label='Number of Bankrupt Agents')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Bankrupt Agents')
    plt.title('Number of Bankrupt Agents Over Time')
    plt.legend()

    # Subplot 2: Difference in Number of Bankrupt Agents
    plt.subplot(5, 2, 2)
    plt.plot(time_steps[1:], diff_bankrupt_agents, label='Difference in Bankrupt Agents', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Difference in Number of Bankrupt Agents')
    plt.title('Difference in Number of Bankrupt Agents Over Time')
    plt.legend()


    # Subplot 3: Cumulative Histogram (in reverse order) of abDifference in Bankrupt Agents in Log-Log scale
    plt.subplot(5, 2, 3)
    plt.hist(np.abs(diff_bankrupt_agents), bins=50, cumulative=-1, log=True, color='green', label='Cumulative Histogram of Diff in Bankrupt Agents', histtype='step', density = 1)
    plt.xscale('log')
    plt.xlabel('Absolute Difference in Number of Bankrupt Agents  (Log Scale)')
    plt.ylabel(' 1 - Cumulative Count (Log Scale)')
    plt.title('Cumulative Histogram of |difference Bankrupt Agents| Over Time')
    plt.legend()


    # Subplot 4: Frequency of Absolute Differences in Number of Bankrupt Agents
    plt.subplot(5, 2, 4)
    # Count the frequency of each difference
    unique_diffs, counts = np.unique(np.abs(diff_bankrupt_agents), return_counts=True)
    # Now create the scatter plot with frequencies
    plt.scatter(unique_diffs, counts)
    plt.xlabel('Absolute Difference in Number of Bankrupt Agents')
    plt.ylabel('Frequency of Occurrence')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Frequency of Absolute Differences in Number of Bankrupt Agents')


    # Subplot 5: Simulated Prices Over Time
    plt.subplot(5, 2, 5)
    plt.plot(time_steps, simulated_prices, label='Simulated Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Simulated Prices Over Time')
    plt.legend()

    # Subplot 6: Number of Nodes Over Time
    plt.subplot(5, 2, 6)
    plt.plot(time_steps, node_population_over_time, label='Number of Nodes')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Nodes')
    plt.title('Number of Nodes Over Time')
    plt.legend()

    # Subplot 7: Number of Links in the Network Over Time
    plt.subplot(5, 2, 7)
    plt.plot(time_steps, links_over_time, label='Number of Links in the Network')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Links')
    plt.title('Number of Links in the Network Over Time')
    plt.legend()

    # Subplot 8: Summed abs(Exposure) stored in links of All Agents Over Time
    plt.subplot(5, 2, 8)
    plt.plot(time_steps, total_abs_exposure_in_edge_weights, label='Total abs Exposure in Links')
    plt.xlabel('Time Step')
    plt.ylabel('Total abs Exposure stored in links')
    plt.title('Summed abs(Exposure) stored in links of All Agents Over Time')
    plt.legend()

    # Subplot 9: Summed Exposure of All Agents Node exposure Over Time
    plt.subplot(5, 2, 9)
    plt.plot(time_steps, exposures_over_time, label='Summed Exposure')
    plt.xlabel('Time Step')
    plt.ylabel('Total Exposure')
    plt.title('Summed Exposure of All Agents Node exposure Over Time')
    plt.legend()

    # Subplot 10: Network Density Over Time
    plt.subplot(5, 2, 10)
    plt.plot(time_steps, network_density_over_time, label='Network Density')
    plt.xlabel('Time Step')
    plt.ylabel('Network Density')
    plt.title('Network Density Over Time')
    plt.legend()


    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()




def analyze_network(G):
    # Initialize DataFrame
    stats_df = pd.DataFrame(index=G.nodes())

    # Centralities
    stats_df['Betweenness Centrality'] = pd.Series(nx.betweenness_centrality(G))
    stats_df['Closeness Centrality'] = pd.Series(nx.closeness_centrality(G))
    stats_df['Eigenvector Centrality'] = pd.Series(nx.eigenvector_centrality(G, max_iter=500))

    # Node Degree (In and Out for Directed Graph)
    if G.is_directed():
        stats_df['In-Degree'] = pd.Series(dict(G.in_degree()))
        stats_df['Out-Degree'] = pd.Series(dict(G.out_degree()))
    else:
        stats_df['Degree'] = pd.Series(dict(G.degree()))

    # Diameter and Average Shortest Path Length
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_shortest_path = nx.average_shortest_path_length(G)
        stats_df['Diameter'] = diameter
        stats_df['Average Shortest Path Length'] = avg_shortest_path
    else:
        print("The graph is not connected. Diameter and Average Shortest Path Length cannot be computed.")

    # Plot Degree Distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 1, 1))
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    return stats_df


def save_results_to_csv(run_results, combination, mode, bankruptcy_mode):
    """
    Saves the results of a simulation run to a CSV file.
    ...
    """
    # Create a DataFrame for the results
    df = pd.DataFrame()
    for i, result in enumerate(run_results):
        num_bankrupt_agents = np.array(result['num_bankrupt_agents_over_time'])

        # Calculate the absolute difference
        abs_diff_bankrupt_agents = np.abs(np.diff(num_bankrupt_agents, prepend=num_bankrupt_agents[0])).tolist()

        df.at[i, 'Run'] = i + 1
        df.at[i, 'Exposures Over Time'] = str(result['abs_exposures_over_time_summed'].tolist())
        df.at[i, 'Number of Bankrupt Agents Over Time'] = str(num_bankrupt_agents.tolist())
        df.at[i, 'Abs Difference of Bankrupt Agents Over Time'] = str(abs_diff_bankrupt_agents)
        df.at[i, 'Simulated Prices'] = str(result['simulated_prices'].tolist())
        df.at[i, 'Links Over Time'] = str(result['links_over_time'].tolist())
        df.at[i, 'Total Absolute Exposure in Edge Weights'] = str(result['total_abs_exposure_in_edge_weights'].tolist())
        df.at[i, 'Node Population Over Time'] = str(result['node_population_over_time'].tolist())

    # Create an initial DataFrame with the combination
    combination_df = pd.DataFrame({'Combination': [str(combination)], 'Run': [None], 'Exposures Over Time': [None],
                                   'Number of Bankrupt Agents Over Time': [None], 'Abs Difference of Bankrupt Agents Over Time': [None],
                                   'Simulated Prices': [None], 'Links Over Time': [None], 'Total Absolute Exposure in Edge Weights': [None],
                                   'Node Population Over Time': [None]})

    # Concatenate the combination row and the results DataFrame
    final_df = pd.concat([combination_df, df], ignore_index=True)

    # Create a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{'_'.join(map(str, combination))}_{mode}_{bankruptcy_mode}_{timestamp}.csv"

    # Save to CSV
    final_df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")


    
def save_all_results_to_csv(all_results, mode, bankruptcy_mode):
    """
    Saves a dictionary of simulation results to a CSV file.
    ...

    Parameters:
    all_results (Dict[Any, List[Dict[str, Any]]]): A dictionary where each key is a combination of
                                                   parameters and each value is a list of result dictionaries.
    Returns:
    None: The function outputs a CSV file and prints the filename.
    """

    # List to store all result rows
    rows = []

    for combination_dict, run_results in all_results.items():
        # Format the combination dictionary into a string
        combination_str = ', '.join([f"{k}={v}" for k, v in combination_dict.items()])

        for i, result in enumerate(run_results):
            # Create a row dictionary, excluding the 'graph' key
            row = {'Combination': combination_str, 'Run': i + 1}
            for key, value in result.items():
                if key != 'graph':
                    if isinstance(value, np.ndarray):  # Check if value is a numpy array
                        row[key] = str(value.tolist())  # Convert numpy array to list and then to string
                    else:
                        row[key] = str(value)
            rows.append(row)

    # Convert list of dictionaries to DataFrame
    all_df = pd.DataFrame(rows)

    # Create a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"simulation_results_{mode}_{bankruptcy_mode}_{timestamp}.csv"

    # Save DataFrame to CSV
    all_df.to_csv(filename, index=False)
    print(f"Saved all results to {filename}")



def calculate_std_diff(timeseries):
    """
    Calculate the standard deviation of the first-order differences of a time series.

    Parameters:
    timeseries (np.array): A time series data in the form of a NumPy array.

    Returns:
    float: The standard deviation of the first-order differences of the time series.
    """
    return np.std(np.diff(timeseries))




def test_bankruptcy_and_exposure_update():
    """# Test check_bankruptcy_and_update_network to verify that the exposure of neighbour node has been updated as expected after one node is bankrupt"""
    # Create a small test graph
    G = nx.DiGraph()
    G.add_node(0, exposure=20)  # Add a node that will go bankrupt
    G.add_node(1, exposure=4)   # Add a node connected to the bankrupt node
    G.add_node(2, exposure=-2)  # Add another unrelated node
    G.add_edge(1, 0, weight=4)  # Link between bankrupt node and node 1
    G.add_edge(0, 1, weight=-4) # Link in the opposite direction
    # Define test parameters
    threshold_v = 10  # Bankruptcy threshold
    delta_price = 1.5 # Price fluctuation

    # Apply the function
    G, num_bankruptcies = check_bankruptcy_and_update_network(G, threshold_v, delta_price)

    # Verify that the bankrupt node has been removed
    assert 0 not in G

    # Verify that the exposure of node 1 has been updated as expected
    expected_exposure = 4 + 4  # Original exposure + weight of the bankrupt node
    print('expected_exposure:', expected_exposure)
    actual_exposure = G.nodes[1]['exposure']
    print('actual_exposure:', actual_exposure)
    assert actual_exposure == expected_exposure

    # Verify the number of bankruptcies
    assert num_bankruptcies == 1

    # Verify that other nodes are unaffected
    assert G.nodes[2]['exposure'] == -2

    print("All tests passed")





def test_maturity_0_and_exposure_update():
    """Test check edge maturity turns to be 0 the exposure of node has been updated as expected"""
    # Create a small test graph
    G = nx.DiGraph()
    G.add_node(0, exposure=5)  # Add a node that will go bankrupt
    G.add_node(1, exposure=4)   # Add a node connected to the bankrupt node
    G.add_node(2, exposure=-2)  # Add another unrelated node
    G.add_edge(1, 0, weight=4, time_to_maturity=0)  # Link between bankrupt node and node 1
    G.add_edge(0, 1, weight=-4, time_to_maturity=0) # Link in the opposite direction
    # Define test parameters
    threshold_v = 15  # Bankruptcy threshold
    delta_price = 1.5 # Price fluctuation

    # Apply the function
    G, num_bankruptcies = check_bankruptcy_and_update_network(G, threshold_v, delta_price)


    # Verify that the exposure of node 0&1 has been updated as expected
    expected_exposure = 5 + (-4)  # Original exposure + weight of the bankrupt node
    print('expected_exposure_node0:', expected_exposure)
    actual_exposure = G.nodes[0]['exposure']
    print('actual_exposure_node0:', actual_exposure)
    assert actual_exposure == expected_exposure   


    expected_exposure = 4 + 4  # Original exposure + weight of the bankrupt node
    print('expected_exposure_node1:', expected_exposure)
    actual_exposure = G.nodes[1]['exposure']
    print('actual_exposure_node1:', actual_exposure)
    assert actual_exposure == expected_exposure


    # Verify that other nodes are unaffected
    assert G.nodes[2]['exposure'] == -2

    print("All tests passed")




def safe_literal_eval(s: str) -> Any:
    """
    Safely evaluates a string containing a Python literal from a basic data type.
    If the string contains valid Python literals, it returns the corresponding value.
    If there's a ValueError or SyntaxError during evaluation, it returns np.nan.

    Parameters:
    - s (str): The string to evaluate.

    Returns:
    - Any: The evaluated Python literal value or np.nan in case of an error.
    """
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan

def calculate_average_list(column: pd.Series) -> pd.Series:
    """
    Calculates the average of lists contained within a pandas Series.

    This function first evaluates string representations of lists in the Series to actual lists.
    Then, it calculates the mean of each list. If an element of the Series is not a list or is empty,
    np.nan is returned for that element.

    Parameters:
    - column (pd.Series): A pandas Series containing string representations of lists.

    Returns:
    - pd.Series: A Series with the calculated averages of the lists.
    """
    return column.apply(safe_literal_eval).apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan)

def calculate_std_list(column: pd.Series) -> pd.Series:
    """
    Calculates the standard deviation of lists contained within a pandas Series.

    Similar to `calculate_average_list`, but calculates the standard deviation of the
    lists instead of the average. If an element of the Series is not a list or is empty,
    np.nan is returned for that element.

    Parameters:
    - column (pd.Series): A pandas Series containing string representations of lists.

    Returns:
    - pd.Series: A Series with the calculated standard deviations of the lists.
    """
    return column.apply(safe_literal_eval).apply(lambda x: np.std(x) if isinstance(x, list) and len(x) > 0 else np.nan)

def calculate_elementwise_division(a: List[float], b: List[float]) -> List[float]:
    """
    Performs element-wise division between two lists of numbers.

    Parameters:
    - a (List[float]): The numerator list.
    - b (List[float]): The denominator list.

    Returns:
    - List[float]: A list containing the result of element-wise division of `a` by `b`.
    """
    return [i/j for i, j in zip(a, b) if j != 0] 

def get_statistics(pattern : str = './200_2000_*.csv'):

    '''
    To get statistics of simulation results to plot.
    '''

    all_results = pd.DataFrame()
    all_stds = pd.DataFrame()

    for filepath in glob.glob(pattern):
        df = pd.read_csv(filepath)

        df['Default Prob'] = df.apply(
        lambda row: calculate_elementwise_division(
            safe_literal_eval(row['Number of Bankrupt Agents Over Time']),
            safe_literal_eval(row['Node Population Over Time'])
        ), axis=1)

        averages_df = pd.DataFrame()
        stds_df = pd.DataFrame()

        for column in df.columns[1:-1]:
            avg = calculate_average_list(df[column])
            std = calculate_std_list(df[column])
            averages_df[column] = avg
            stds_df[column] = std


        averages_df['Default Prob'] = df['Default Prob'].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan)
        stds_df['Default Prob'] = df['Default Prob'].apply(lambda x: np.std(x) if isinstance(x, list) and len(x) > 0 else np.nan)

        averages = averages_df.mean()
        stds = stds_df.mean()

        averages_df_overall = pd.DataFrame([averages])
        stds_df_overall = pd.DataFrame([stds])

        filename = os.path.basename(filepath).replace('.csv', '')
        parts = filename.split('_')
        N_agents = int(parts[0])
        num_steps = int(parts[1])
        delta_t = float(parts[2])
        sigma_exposure_node = float(parts[3])
        sigma_intrestrate = float(parts[4])
        threshold_v = float(parts[5])
        linking_threshold = float(parts[6])

        para_df = pd.DataFrame({
        'N_agents': [N_agents],
        'num_steps': [num_steps],
        'delta_t': [delta_t],
        'sigma_exposure_node': [sigma_exposure_node],
        'sigma_intrestrate': [sigma_intrestrate],
        'threshold_v': [threshold_v],
        'linking_threshold': [linking_threshold]
    })

        temp_df = pd.concat([averages_df_overall, para_df], axis=1)

        all_results = pd.concat([all_results, temp_df], ignore_index=True)
        all_stds = pd.concat([all_stds, stds_df_overall], ignore_index=True)

    return all_results, all_stds



def plot_simu_2D_results(pattern : str = './200_2000_*.csv', x_axis_name : str = 'sigma_intrestrate',
y_axis_name : str = 'Default Prob'):

    all_results, all_stds = get_statistics(pattern = pattern)
    plt.figure(figsize=(10, 6))

    plt.scatter(all_results[x_axis_name], all_results[y_axis_name])

    for x, y, yerr in zip(all_results[x_axis_name], all_results[y_axis_name], all_stds[y_axis_name]):
        plt.vlines(x, y - yerr, y + yerr)
        plt.hlines(y - yerr, x - 0.5, x + 0.5)
        plt.hlines(y + yerr, x - 0.5, x + 0.5)

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(f'{x_axis_name} vs. {y_axis_name}', fontsize=16)

    plt.show()


def plot_simu_3D_results(pattern : str = './200_2000_*.csv', x_axis_name : str = 'sigma_intrestrate',
y_axis_name : str = 'Total Absolute Exposure in Edge Weights', z_axis_name : str = 'Default Prob'):

    all_results, _ = get_statistics(pattern = pattern)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(all_results[x_axis_name], all_results[y_axis_name], all_results[z_axis_name])

    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_zlabel(z_axis_name)
    ax.set_title(f'{x_axis_name} vs.{y_axis_name} vs. {z_axis_name}', fontdict={'fontsize': 14})

    plt.show()

def get_files_with_parameters(directory_path: str, input_parameters: str) -> List[pd.DataFrame]:
    """
    Scans a directory for files that match given input parameters and reads them into pandas DataFrames.
    
    The function expects files to be named with a pattern that includes parameters and a timestamp, separated by underscores. 
    It filters files based on whether the input parameters are found in the file's parameters section (excluding the timestamp).
    
    Parameters:
    - directory_path (str): The path to the directory containing the files to be scanned.
    - input_parameters (str): The parameters to filter the files by.
    
    Returns:
    - List[pd.DataFrame]: A list of pandas DataFrames corresponding to each file that matched the input parameters.
    
    Notes:
    - Files are expected to be CSVs.
    - The file naming convention is assumed to be "<sequence>_<parameters>_<timestamp>.csv".
    """
    
    # Read all files in the directory
    all_files = os.listdir(directory_path)
    print('All files in directory:', all_files)

    # List to hold DataFrames of files that match the input parameters
    matching_files_data = []

    for filename in all_files:
        # Split the filename on the first underscore to separate the sequence number from the rest
        parts = filename.split('_', 1)
        if len(parts) == 2:
            # Split the rest of the filename on the last underscore to isolate parameters from the timestamp
            prefix, parameters_with_timestamp = parts
            parameters = '_'.join(parameters_with_timestamp.split('_')[:-1])
            
            # Check if the file parameters match the input parameters
            if input_parameters in parameters:
                # Construct the full path to the file
                file_path = os.path.join(directory_path, filename)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                # Add the DataFrame to the list of matching files
                matching_files_data.append(df)

    return matching_files_data
def probability_of_default(data_array: List[float], threshold: float) -> float:
    """
    Calculates the probability of values in the data_array exceeding a given threshold.
    
    :param data_array: List of numeric values representing data points.
    :param threshold: A numeric threshold for determining default.
    :return: Probability (float) of a data point exceeding the threshold.
    """
    # Calculate the proportion of data points exceeding the threshold
    return sum([1 if x > threshold else 0 for x in data_array]) / len(data_array)

def convert_string_to_list(string):
    # Remove the square brackets
    string = string.strip('[]')
    # Split the string on commas
    string = string.split(',')
    # Convert each string in the list to a float
    string = [float(s) for s in string]
    return string

def select_variable_given_list_of_dfs(selected_dfs: List[pd.DataFrame], variable_name: str) -> List[Union[pd.Series, List]]:
    """
    Selects and returns a list of Series or lists corresponding to a specified variable
    from each DataFrame in the given list of DataFrames, after dropping the first row of each DataFrame.
    
    Parameters:
    - selected_dfs (List[pd.DataFrame]): A list of pandas DataFrames from which the variable is to be selected.
    - variable_name (str): The name of the variable (column) to be selected from each DataFrame.
    
    Returns:
    - List[Union[pd.Series, List]]: A list containing the pandas Series or lists for the specified variable
      from each DataFrame in `selected_dfs`, excluding the first row of each DataFrame. If a DataFrame does not
      contain the specified variable, it is skipped, and a message is printed to indicate its absence.
      
    Note:
    - This function assumes that the variable column might contain data of mixed types (e.g., numbers, lists, or arrays),
      hence no data type conversion is performed.
    - If the specified variable is not found in a DataFrame, a warning message is printed, and that DataFrame is skipped.
    """
    
    # List to hold Series or lists of the selected variable from each DataFrame
    list_of_series_or_lists = []

    for df in selected_dfs:
        # Check if the variable_name column exists to avoid KeyError
        if variable_name in df.columns:
            # Drop the first row from the DataFrame
            df = df.drop(df.index[0])

            # Extract the variable column as it may contain NaNs or mixed types
            variable_series = df[variable_name]
            
            # Append the extracted column to the list
            list_of_series_or_lists.append(variable_series)
        else:
            # Print a message if the specified variable is not found in the DataFrame
            print(f"The variable '{variable_name}' is not in the DataFrame.")
    return list_of_series_or_lists


def plot_bankruptcy_analysis_multiple_arrays(selected_files, titles, param_caption):
    """
    Plots the bankruptcy analysis with three subplots for each file in data_array.
    :param selected_files: List of file paths to read bankruptcy data from.
    :param titles: Titles for each file to be used in plots.
    :param param_caption: Dictionary of parameters to display as figure caption.
    """
    bankruptcy_dfs = select_variable_given_list_of_dfs(selected_files, 'Abs Difference of Bankrupt Agents Over Time')
    exposure_dfs = select_variable_given_list_of_dfs(selected_files, 'Total Absolute Exposure in Edge Weights')
    
    for file_index, (bankruptcy_df, exposure_df) in enumerate(zip(bankruptcy_dfs, exposure_dfs)):
        plt.figure(figsize=(10, 24))
        all_runs_results = []
        
        for run in range(bankruptcy_df.shape[0]): 
            diff_bankrupt_agents = convert_string_to_list(bankruptcy_df.iloc[run])
            total_abs_exposure_in_edge_weights = convert_string_to_list(exposure_df.iloc[run])
            time_steps = np.arange(len(diff_bankrupt_agents))



            # Analyzing time series for each run
            print(f"Results Summary for {titles[file_index]}, run {run + 1}:")
            analysis_results_df = analyze_time_series(diff_bankrupt_agents, total_abs_exposure_in_edge_weights)
            # print residual test
            print('correlation', analysis_results_df['correlation'])
            print('Portmaneau test for autocorrelation of residuals p value:', analysis_results_df['var_residuals_test'][0])
            if analysis_results_df['var_residuals_test'][0] < 0.05:
                print('Portmaneau test for autocorrelation of residuals rejects H0 that there is serial auto correlation')
            print('Augmented Dickey Fuller test abs difference number of difference of number of Agents:', analysis_results_df['adf_pvalue_ts1'][0])
            if analysis_results_df['adf_pvalue_ts1'][0] < 0.05:
                print('Augmented Dickey Fuller test abs difference number of difference of number of Agents rejects H0 that there is no unit root, so the timeseries is stationary')
            print('Augmented Dickey Fuller test abs difference number of Total abs exposure stored in links:', analysis_results_df['adf_pvalue_ts1'][0])
            if analysis_results_df['adf_pvalue_ts1'][0] < 0.05:
                print('Augmented Dickey Fuller test abs difference number of difference of number of Agents rejects H0 that there is no unit root, so the timeseries is stationary')
            print('Rank of cointegrating matrix (number of cointegrating relationships):', analysis_results_df['coint_rank'])
            print('VAR lag order:', analysis_results_df['var_lag_order'])
            print('BIC:', analysis_results_df['var_bic'])
            analysis_results_dict = analysis_results_df.iloc[0].to_dict()  # Convert the DataFrame row to a dictionary
            analysis_results_dict['run'] = run + 1  # Adding run identifier for tracking
            all_runs_results.append(analysis_results_dict)

            # Generate a random color for each run
            random_color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            # Plotting code
            # Subplot 1: Difference in Number of Bankrupt Agents
            plt.subplot(4, 1, 1)
            plt.plot(time_steps, diff_bankrupt_agents, label=f'Run {run+1}', color=random_color)
            plt.xlabel('Time Step')
            plt.ylabel('Bankrupt Nodes per timestep')
            plt.title(f'N = {titles[file_index]} - Bankrupt Nodes per timestep')
            plt.legend()

            # Subplot 2: Cumulative Histogram of Absolute Difference in Bankrupt Agents
            plt.subplot(4, 1, 2)
            plt.hist(np.abs(diff_bankrupt_agents), bins=50, cumulative=-1, log=True, color=random_color, label=f'Run {run+1}', histtype='step', density=True)
            plt.xscale('log')
            plt.xlabel('Bankrupt Nodes per timestep (Log Scale)')
            plt.ylabel('1 - Cumulative Count (Log Scale)')
            plt.title(f'N = {titles[file_index]} - Cumulative Histogram of |Difference in Bankrupt Agents| Over Time')
            plt.legend()

            # Subplot 3: Frequency of Absolute Differences in Number of Bankrupt Agents
            plt.subplot(4, 1, 3)
            unique_diffs, counts = np.unique(np.abs(diff_bankrupt_agents), return_counts=True)
            plt.scatter(unique_diffs, counts, label=f'Run {run+1}', color=random_color)
            plt.xlabel('Absolute Difference in Number of Bankrupt Agents')
            plt.ylabel('Frequency of Occurrence')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'N = {titles[file_index]} - Frequency of Absolute Differences in Number of Bankrupt Agents')
            plt.legend()

            param_caption_str = ', '.join([f"{key}={value}" for key, value in param_caption.items()])
            plt.figtext(0.5, -0.05, f"Parameters: {param_caption_str}", wrap=True, horizontalalignment='center', fontsize=10)
            plt.tight_layout(pad=3.0)

            # Subplot 4: Summed abs(Exposure) stored in links of All Agents Over Time
            plt.subplot(4, 1, 4)
            plt.plot(time_steps, total_abs_exposure_in_edge_weights, label=f'Run {run+1}', color=random_color)
            plt.xlabel('Time Step')
            plt.ylabel(f'Total |Exposure| stored in links')
            plt.title(f'{titles[file_index]} Summed |Exposure| stored in links of All Agents Over Time')
            plt.legend()

        plt.show()


def perform_regression(dependent_var, *independent_vars):
    """
    Perform linear regression (simple or multiple) based on the number of independent variables provided.

    Parameters:
    dependent_var (array-like): The dependent variable.
    independent_vars (variable number of array-like): The independent variable(s).

    Returns:
    RegressionResults: The results of the regression analysis.
    """
    # Combine independent variables into a DataFrame
    data = pd.DataFrame({f'var{i}': var for i, var in enumerate(independent_vars, start=1)})

    # Add the dependent variable
    data['dependent'] = dependent_var

    # Define the formula
    independent = ' + '.join(data.columns[:-1])
    formula = f'dependent ~ {independent}'

    # Fit the model
    model = sm.OLS.from_formula(formula, data).fit()

    return model.summary()


def plot_default_size_dot(data_array_num: List[Any], data_array_diff: List[str], titles: List[str], system_sizes: List[int], default_consideration_threshold: float = 0.1) -> None:
    """
    Plot the default probability per system size with scatter and error bars indicating standard deviation.

    Parameters:
        data_array_num (List[Any]): A list containing numerical data arrays.
        data_array_diff (List[str]): A list of strings representing differences in data, e.g., number of bankrupt agents per run.
        titles (List[str]): Titles corresponding to each data array, used for labeling or categorization.
        system_sizes (List[int]): A list of integers representing different system sizes to be analyzed.
        default_consideration_threshold (float, optional): A threshold value used in calculating default probability. Defaults to 0.1.
    
    Returns:
        None: This function does not return any value but plots a figure with matplotlib.
    """
    plt.figure(figsize=(10, 6))

    # Data structure to hold default probabilities for each system size
    default_probs_data: Dict[int, List[float]] = {size: [] for size in system_sizes}

    # Lists to collect data for scatter plot
    scatter_data_x: List[int] = []
    scatter_data_y: List[float] = []

    # Process data to calculate default probabilities
    for file_index, (num_data, diff_data, title, size) in enumerate(zip(data_array_num, data_array_diff, titles, system_sizes)):
        for run_index, diff_str in enumerate(diff_data):
            diff_bankrupt_agents = convert_string_to_list(diff_str)
            default_probability, _ = calculate_default_probability(diff_bankrupt_agents, default_consideration_threshold, size)
            default_probs_data[size].append(default_probability)

            # Collect data for scatter plot
            scatter_data_x.append(size)
            scatter_data_y.append(default_probability)

    # Group default probabilities by system size for statistical analysis
    grouped_default_probs = [default_probs_data[size] for size in system_sizes]

    # Performing the statistical test and printing the result
    test_result = compare_group_means(*grouped_default_probs)
    print(test_result)  # This will print the result to the notebook

    # Plot individual runs as scatter points
    plt.scatter(scatter_data_x, scatter_data_y, s=50, alpha=0.75, edgecolors='w', linewidths=0.5)

    # Plot error bars for mean and standard deviation
    for size in system_sizes:
        mean_prob = np.mean(default_probs_data[size])
        std_prob = np.std(default_probs_data[size])
        plt.errorbar(size, mean_prob, yerr=std_prob, fmt='o', color='red', capsize=5, label='Mean with 1 Std.' if size == system_sizes[0] else "")

    # Formatting the plot
    plt.xlabel('System Size')
    plt.ylabel('Default Probability')
    plt.title('Default Probability per System Size with Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_default_size(data_array_num_model, data_array_diff_model, data_array_num_paper, data_array_diff_paper, titles, system_sizes, default_consideration_threshold, param_caption):
    all_data_model = {}
    all_data_paper = {}
    regression_results = []

    # Process model data
    for system_size, num_data, diff_data in zip(system_sizes, data_array_num_model, data_array_diff_model):
        average_default_sizes = []
        for num_str, diff_str in zip(num_data, diff_data):
            adjusted_bankrupt_agents = adjust_bankrupt_agents(num_str, diff_str)
            average_default_size = np.mean(adjusted_bankrupt_agents)  # Compute the average default size
            average_default_sizes.append(average_default_size)
        all_data_model[system_size] = average_default_sizes

    # Process paper data
    for system_size, num_data, diff_data in zip(system_sizes, data_array_num_paper, data_array_diff_paper):
        average_default_sizes = []
        for num_str, diff_str in zip(num_data, diff_data):
            adjusted_bankrupt_agents = adjust_bankrupt_agents(num_str, diff_str)
            average_default_size = np.mean(adjusted_bankrupt_agents)  # Compute the average default size
            average_default_sizes.append(average_default_size)
        all_data_paper[system_size] = average_default_sizes

    # Plotting box plots
    plt.figure(figsize=(15, 9))
    positions_model = [i*2 for i, _ in enumerate(system_sizes)]
    positions_paper = [i*2 + 1 for i, _ in enumerate(system_sizes)]

    box_data_model = [all_data_model[size] for size in system_sizes]
    box_data_paper = [all_data_paper[size] for size in system_sizes]

    plt.boxplot(box_data_model, positions=positions_model, widths=0.6, patch_artist=True, boxprops=dict(facecolor='blue'))
    plt.boxplot(box_data_paper, positions=positions_paper, widths=0.6, patch_artist=True, boxprops=dict(facecolor='orange'))

    # Adding titles and labels
    plt.xticks([i*2 + 0.5 for i, _ in enumerate(system_sizes)], [f'N={size}' for size in system_sizes])
    plt.xlabel('System Size')
    plt.ylabel('Average Default Size')
    plt.title('Comparison of Average Default Size per System Size')

    # Create custom legends
    blue_patch = mpatches.Patch(color='blue', label='Our Model')
    orange_patch = mpatches.Patch(color='orange', label='Referenced Paper')
    plt.legend(handles=[blue_patch, orange_patch], loc='upper right')

    plt.grid(True)

    # Parameters caption
    param_caption_str = ', '.join(f"{key}: {value}" for key, value in param_caption.items())
    plt.figtext(0.5, -0.05, f"Parameters: {param_caption_str}", wrap=True, horizontalalignment='center', fontsize=10)

    # Statistical comparison for each system size
    list_of_means_model = []
    list_of_means_paper = []
    for size in system_sizes:
        model_data = all_data_model[size]
        average_size_calc_model =  np.mean(model_data)
        list_of_means_model.append(average_size_calc_model)
        paper_data = all_data_paper[size]
        average_size_calc_paper = np.mean(paper_data)
        list_of_means_paper.append(average_size_calc_paper)


    # Perform regression analysis
    print(list_of_means_model)
    print(system_size)
    print("Our Model")
    print(perform_regression(list_of_means_model, system_sizes))
    print("Drupsteen Model")
    print(perform_regression(list_of_means_paper, system_sizes))



    print(model_data[0])
    for size, result in regression_results:
        print(f"System Size: {size}, Regression Result:\n{result}")

    plt.show()

def compare_group_means(*groups):
    """
    Compare means of multiple groups using ANOVA or Kruskal-Wallis test.
    Includes a conclusion based on the hypothesis test.
    """
    # Check for zero variance within each group
    if any(np.ptp(group) == 0 for group in groups):
        print("One or more groups have zero variance, performing non-parametric Kruskal-Wallis test.")
        kruskal_result = stats.kruskal(*groups)
        conclusion = "There is no significant difference between the groups." if kruskal_result.pvalue > 0.05 else "There is a significant difference between the groups."
        return f"Kruskal-Wallis Test Result:\nH-statistic: {kruskal_result.statistic}\nP-value: {kruskal_result.pvalue}\nConclusion: {conclusion}"

    # Create a DataFrame to hold all groups data
    group_labels = [f'group{i+1}' for i in range(len(groups))]
    data_values = []
    data_groups = []
    for i, group in enumerate(groups):
        data_values.extend(group)
        data_groups.extend([group_labels[i]] * len(group))
    
    df = pd.DataFrame({'value': data_values, 'group': data_groups})

    # Checking for normality and homogeneity of variances
    normality_p_values = [stats.shapiro(df[df['group'] == group]['value']).pvalue for group in group_labels]
    normal = all(p > 0.05 for p in normality_p_values)
    homogeneity_p_value = stats.levene(*groups).pvalue
    homogeneous = homogeneity_p_value > 0.05

    if normal and homogeneous:
        print("The data is normal and homogeneous, performing ANOVA.")
        # Perform ANOVA
        model = ols('value ~ C(group)', data=df).fit()
        anova_result = sm.stats.anova_lm(model, typ=2)
        conclusion = "There is no significant difference between the groups." if anova_result['PR(>F)'].iloc[0] > 0.05 else "There is a significant difference between the groups."
        return f"ANOVA Result:\n{anova_result}\nConclusion: {conclusion}"
    else:
        print("The data is not normal and/or not homogeneous, performing non-parametric Kruskal-Wallis test.")
        # Perform Kruskal-Wallis Test
        kruskal_result = stats.kruskal(*groups)
        conclusion = "There is no significant difference between the groups." if kruskal_result.pvalue > 0.05 else "There is a significant difference between the groups."
        return f"Kruskal-Wallis Test Result:\nH-statistic: {kruskal_result.statistic}\nP-value: {kruskal_result.pvalue}\nConclusion: {conclusion}"


def perform_regression_inc_interaction(dependent_var, *independent_vars):
    """
    Perform linear regression (simple or multiple) based on the number of independent variables provided,
    including interaction terms between all independent variables.

    Parameters:
    dependent_var (array-like): The dependent variable.
    independent_vars (variable number of array-like): The independent variable(s).

    Returns:
    RegressionResults: The results of the regression analysis, including interaction terms.
    """
    # Combine independent variables into a DataFrame
    data = pd.DataFrame({f'var{i}': var for i, var in enumerate(independent_vars, start=1)})

    # Add the dependent variable
    data['dependent'] = dependent_var

    # Define the formula with interaction terms
    # Generate all possible combinations of interaction terms
    independent_vars_names = data.columns[:-1]
    interaction_terms = ' + '.join([f'{var1}:{var2}' for i, var1 in enumerate(independent_vars_names) 
                                     for var2 in independent_vars_names[i+1:]])

    # Combine the main effects and interaction terms
    if interaction_terms:
        formula = f'dependent ~ {" + ".join(independent_vars_names)} + {interaction_terms}'
    else:
        formula = f'dependent ~ {" + ".join(independent_vars_names)}'

    # Fit the model
    model = sm.OLS.from_formula(formula, data).fit()

    return model.summary()

def fit_power_law_array(data_array, titles):
    """
    Fit data from multiple arrays to a power-law distribution and compare it with an exponential distribution.
    Includes an aggregated p-value using Stouffer's Z-method in its own row below the runs it aggregates.

    Parameters:
    data_array (List[pd.DataFrame]): List of pandas DataFrames with data to be fitted.
    titles (List[str]): List of titles corresponding to each DataFrame in data_array.

    Returns:
    pd.DataFrame: A DataFrame containing individual run data and aggregated p-values for each title.
    """
    results_list = []

    for file_index, file in enumerate(data_array):
        p_values = []
        for run in range(file.shape[0]):
            run_data = file.iloc[run]
            run_data = convert_string_to_list(run_data)

            # Fit the data to a power-law distribution
            results = powerlaw.Fit(run_data)
            alpha = results.power_law.alpha
            R, p = results.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            p_values.append(p)


            # Format p-value with significance levels
            p_value_formatted = f"{np.round(p, 4)}"
            if p < 0.001:
                p_value_formatted += "***"  # Highly significant
            elif p < 0.01:
                p_value_formatted += "**"   # Very significant
            elif p < 0.05:
                p_value_formatted += "*"    # Significant

            # Add results to list
            results_list.append({
                'N agents (Nodes)': titles[file_index],
                'run': run + 1,
                'alpha exponent': alpha,
                'likelihood ratio': R,
                'p-value': p_value_formatted
            })

            # Aggregate p-values for this title using Stouffer's Z-method from statsmodels
            combined_test_stat, aggregated_p_value = combine_pvalues(p_values, method='stouffer')


            # Format aggregated_p_value p-value with significance levels
            aggregated_p_value_formatted = f"{np.round(aggregated_p_value, 4)}"
            if aggregated_p_value < 0.001:
                aggregated_p_value_formatted += "***"  # Highly significant
            elif aggregated_p_value < 0.01:
                aggregated_p_value_formatted += "**"   # Very significant
            elif aggregated_p_value < 0.05:
                aggregated_p_value_formatted += "*"    # Significant



        results_list.append({
            'N agents (Nodes)': '',
            'run': 'Stouffer’s p-value',
            'alpha exponent': '',
            'likelihood ratio': '',
            'p-value': aggregated_p_value_formatted
        })

    # Create DataFrame for all results
    results_df = pd.DataFrame(results_list)
    return results_df


def perform_one_way_anova(dataframe, dependent_var, independent_var):
    """
    Perform a one-way ANOVA test.

    Parameters:
    dataframe (pandas.DataFrame): The dataset containing the variables.
    dependent_var (str): The name of the dependent variable (numeric).
    independent_var (str): The name of the independent variable (categorical).

    Returns:
    ANOVAResults: The results of the ANOVA test.
    """
    # Fit the model
    model = ols(f'{dependent_var} ~ C({independent_var})', data=dataframe).fit()

    # Perform ANOVA
    anova_results = sm.stats.anova_lm(model, typ=2)
    return anova_results


def stress_vs_avalanche_prob_vs_volatility(directory:str = './', threshold = 10, mode:str = 'sigma_interest_rates'):
    """
    Analyzes and visualizes the relationship between stress, avalanche probability, and volatility based on grouped financial data.

    This function groups files by a specified mode, calculates the average default probability and total absolute exposure in edge weights,
    and then visualizes these against a volatility metric defined by either 'sigma_interest_rates' or 'sigma_exposure_node'.

    Parameters:
    - directory (str): The path to the directory containing the files to be analyzed. Defaults to './'.
    - threshold (int): The threshold used to calculate the default probability. Defaults to 10.
    - mode (str): The mode for grouping files and analyzing data. Can be 'sigma_interest_rates' or 'sigma_exposure_node'. Defaults to 'sigma_interest_rates'.

    The function does not return any value but outputs a 3D plot showing the relationship between the variables and prints a model summary.
    """

    # Group files based on the specified mode (e.g., 'sigma_interest_rates' or 'sigma_exposure_node').
    groups = group_files(directory, mode)

    # Iterate through each group to process the files
    for key, value in groups.items():
        print("Key:", key)  # Debugging print to check the current group key

        # Initialize lists to store calculated averages for each file in the group
        bankrupt_nodes_avg_list = []
        total_edge_exposures = []
        filenames = value['filenames']

        # Process each file in the current group
        for filename in filenames:
            full_path = os.path.join(directory, filename)  # Create the full file path
            df = pd.read_csv(full_path)  # Read the CSV file into a DataFrame

            # Initialize a list to store bankruptcy node calculations for each row in the DataFrame
            bankrupt_nodes = []
            # Iterate through each row in the DataFrame to calculate bankruptcy nodes and exposures
            for i in range(len(df)):
                # Skip rows where necessary data is missing
                if pd.isna(df['Abs Difference of Bankrupt Agents Over Time'].iloc[i]) or pd.isna(df['Number of Bankrupt Agents Over Time'].iloc[i]):
                    continue
                else: 
                    # Calculate the default probability and add it to the list
                    bankrupt_node = calculate_probability(ast.literal_eval(df['Abs Difference of Bankrupt Agents Over Time'].iloc[i]), threshold=threshold)
                    bankrupt_nodes.append(bankrupt_node)

            # Calculate the average of bankruptcy nodes and add it to the list
            bankrupt_nodes_avg = np.mean(bankrupt_nodes)
            bankrupt_nodes_avg_list.append(bankrupt_nodes_avg)

            # Calculate the average total exposure for each file and add it to the list
            exposures = [ast.literal_eval(x) for x in df['Total Absolute Exposure in Edge Weights'] if not pd.isna(x)]
            exposures_avg = [np.mean(x) for x in exposures]
            total_edge_exposure_avg = np.mean(exposures_avg)
            total_edge_exposures.append(total_edge_exposure_avg)

        # Retrieve volatility measures based on the specified mode
        if mode == 'sigma_interest_rates':
            sigmas = value['sigma_interest_rates']
        elif mode == 'sigma_exposure_node':
            sigmas = value['sigma_exposure_node']

        # Convert sigma values to float for analysis
        sigmas = [float(sigma) for sigma in sigmas]

        # Combine and sort the calculated values for plotting
        combined = sorted(zip(sigmas, bankrupt_nodes_avg_list, total_edge_exposures))
        sigmas_sorted, bankrupt_nodes_list_sorted, total_edge_exposures_sorted = zip(*combined)

        # Set up the plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D surface plot
        surf = ax.plot_trisurf(sigmas_sorted, total_edge_exposures_sorted, bankrupt_nodes_list_sorted)

        # Label the axes and set the title
        ax.set_xlabel(mode.replace('_', ' ').title())
        ax.set_ylabel('Total Absolute Exposure in Edge Weights')
        ax.set_zlabel('Default Prob')
        ax.set_title('Stress vs. Avalanche Prob vs. Volatility', fontdict={'fontsize': 14})
        ax.view_init(elev=30, azim=60)

        # Perform regression analysis on the sorted lists and print the summary
        model_summary = perform_regression_inc_interaction(bankrupt_nodes_list_sorted, sigmas_sorted, total_edge_exposures_sorted)
        print(model_summary)

        # Display the plot
        plt.show()


def plot_avalanche_prob_vs_volatility_combined(directory: str = './', threshold: int = 15, mode: str = 'sigma_exposure_node') -> None:
    """
    Plot the probability of avalanche (default) versus volatility for grouped data.

    This function reads financial data from CSV files, calculates the probability of default
    based on the absolute difference of bankrupt agents over time, and plots this probability
    against the volatility (sigma exposure) of the nodes.

    Parameters:
    - directory (str): The directory containing the CSV files.
    - threshold (int): The threshold value used to calculate the probability of default.
    - mode (str): The mode to use for grouping files. It also specifies the x-axis label of the plot.

    Returns:
    - None: This function does not return a value but displays a plot.
    """
    
    # Group files based on the provided mode
    groups = group_files(directory, mode)
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    colors = ['orange', 'green']  # Colors for different groups
    color_index = 0

    legend_handles = []  # For custom legend
    offset = -0.8  # Offset for side-by-side boxplots

    # Process each group of files
    for key, value in groups.items():       
        bankrupt_nodes_list = []
        filenames = value['filenames']
        
        # Read and process each file
        for filename in filenames:
            full_path = os.path.join(directory, filename)
            df = pd.read_csv(full_path)
            bankrupt_nodes = []
            
            # Calculate the probability of default for each time point
            for i in range(len(df)):
                if pd.isna(df['Abs Difference of Bankrupt Agents Over Time'].iloc[i]) or pd.isna(df['Number of Bankrupt Agents Over Time'].iloc[i]):
                    continue
                else: 
                    bankrupt_node = calculate_probability(ast.literal_eval(df['Abs Difference of Bankrupt Agents Over Time'].iloc[i]), threshold=threshold)
                    bankrupt_nodes.append(bankrupt_node)
            bankrupt_nodes_list.append(bankrupt_nodes)
            
        # Sort data based on sigma values for plotting
        sigmas = value['sigma_exposure_node']
        combined = sorted(zip(sigmas, bankrupt_nodes_list))
        sigmas_sorted, bankrupt_nodes_list_sorted = zip(*combined)

        base_positions = np.arange(len(bankrupt_nodes_list_sorted)) * 2
        positions = base_positions + (color_index * offset - offset / 2)

        color_index += 1
        box_color = colors[color_index % len(colors)]
        
        # Create boxplot for each group
        boxplots = plt.boxplot(bankrupt_nodes_list_sorted, positions=positions, labels=sigmas_sorted, patch_artist=True,
                               boxprops=dict(facecolor=box_color, color=box_color),
                               medianprops=dict(color=box_color),
                               whiskerprops=dict(color=box_color),
                               capprops=dict(color=box_color),
                               flierprops=dict(markeredgecolor=box_color))

        for patch in boxplots['boxes']:
            patch.set_facecolor(box_color)
        
        # Add legend handle for this group
        legend_handles.append(plt.Line2D([0], [0], color=box_color, label=key[8:]))


    
    
    # Additional plot formatting
    plt.figtext(0.5, 0, f"Parameters: {key}", wrap=True, horizontalalignment='center', fontsize=12)
    plt.xlabel(mode.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Default Prob', fontsize=14)
    plt.title('Avalanche Prob vs. Volatility', fontsize=16)
    plt.legend(handles=legend_handles, loc='upper left')
    plt.xticks(base_positions, labels=sigmas_sorted)
    #plt.tight_layout()
    plt.show()

def get_statistics_vary_threshold_v(pattern: str = './200_2000_*.csv') -> Dict[float, List[float]]:
    """
    Aggregates the average absolute difference of bankrupt agents over time from CSV files,
    organized by a threshold value extracted from the filenames.
    
    Parameters:
    - pattern (str): The glob pattern to match filenames in the current directory.
                     Defaults to './200_2000_*.csv', targeting specific CSV files.
    
    Returns:
    - Dict[float, List[float]]: A dictionary mapping each threshold value (float) to a list
                                of average absolute differences (float) computed from the
                                CSV files that match the threshold value.
    """
    # Dictionary to hold the average absolute difference per threshold
    abs_diff_avg_per_run_per_threshold: Dict[float, List[float]] = {}

    # Loop through files that match the given pattern
    for filepath in glob.glob(pattern):
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath).replace('.csv', '')
        parts = filename.split('_')
        threshold_v = float(parts[5])  # Assuming the threshold value is at index 5
        abs_diff_avg = calculate_average_list(df['Abs Difference of Bankrupt Agents Over Time'])
        
        # Initialize the list for this threshold if it's not already present
        if threshold_v not in abs_diff_avg_per_run_per_threshold:
            abs_diff_avg_per_run_per_threshold[threshold_v] = []
        abs_diff_avg_per_run_per_threshold[threshold_v].append(abs_diff_avg)

    return abs_diff_avg_per_run_per_threshold
