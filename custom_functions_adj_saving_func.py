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
from typing import Dict, List, Any, Tuple
import doctest
import pandas as pd
import powerlaw
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random


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

def draw_graph_with_edge_weights(G, pos=None, node_size=700, node_color='skyblue', font_size=14, 
                                 font_weight='bold', arrowstyle='-|>', arrowsize=20, width=2, 
                                 edge_precision=3):
    """
    Draw a directed graph with edge weights rounded to a specified precision.

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

    Returns:
    None: Draws the graph with matplotlib.


    Note: mind that this picture is not entirely correct, as the weights are able to carry different exposure values depending on the direction of the edge while only 1 is displayed.
    """

    if pos is None:
        pos = nx.spring_layout(G)  # positions for all nodes

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle=arrowstyle, 
                           arrowsize=arrowsize, width=width)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight=font_weight)

    # Draw edge weights with specified precision
    edge_labels = {e: f'{w:.{edge_precision}f}' for e, w in nx.get_edge_attributes(G, 'weight').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')  # Turn off the axis
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
    Saves the results of a simulation run to a CSV file, including parameter settings in the first row.
    ...

    Parameters:
    run_results (list of dicts): Results of the simulation run.
    combination (tuple): Parameter combination for the simulation run.
    mode (str): The mode of the simulation.
    bankruptcy_mode (str): The bankruptcy mode used in the simulation.
    """
    # Create a DataFrame from the results
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
    all_results (Dict[tuple, List[Dict[str, Any]]]): A dictionary where each key is a combination of
                                                    parameters (as a tuple) and each value is a list of 
                                                    result dictionaries.

    Returns:
    None: The function outputs a CSV file and prints the filename.
    """

    # List to store all result rows
    rows = []

    for combination, run_results in all_results.items():
        # Convert tuple to string, joining elements with an underscore
        combination_str = '_'.join(map(str, combination))

        for i, result in enumerate(run_results):
            # Create a row dictionary, excluding the 'graph' key
            row = {'Combination': combination_str, 'Run': i + 1}
            row.update({key: str(value) for key, value in result.items() if key != 'graph'})
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

def get_statistics(pattern : str = './200_2000_*.csv'):

    '''
    To get statistics of simulation results to plot.
    '''

    def safe_literal_eval(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return np.nan

    def calculate_average_list(column):
        return column.apply(safe_literal_eval).apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan)

    def calculate_std_list(column):
        return column.apply(safe_literal_eval).apply(lambda x: np.std(x) if isinstance(x, list) and len(x) > 0 else np.nan)

    def calculate_elementwise_division(a, b):
        return [i/j for i, j in zip(a, b)]


    all_results = pd.DataFrame()
    all_stds = pd.DataFrame()

    for filepath in glob.glob(pattern):
        df = pd.read_csv(filepath)

        df['Default Prob'] = df.apply(
        lambda row: calculate_elementwise_division(
            safe_literal_eval(row['Number of Bankrupt Agents Over Time']),
            safe_literal_eval(row['Node Population Over Time'])
        ), axis=1)

        # print(df['Default Prob'])

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