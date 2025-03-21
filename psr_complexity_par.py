import pdat
import os
import numpy as np
np.bool = bool
import antropy as ant
import matplotlib.pyplot as plt
import nolds
from concurrent.futures import ProcessPoolExecutor

def read_psrfits_data(file_path):
    """
    Reads PSRFITS data from the given file path.
    """
    psrfit_data = pdat.psrfits(file_path)
    data_all = psrfit_data[2].read()
    PSR_name = data_all[0][0].decode('utf-8').strip()
    pulsar_name = PSR_name.split()[1]  # Extract pulsar name
    del data_all
    data_all = psrfit_data[4].read()
    return psrfit_data, pulsar_name, data_all

def process_data_block(data_all):
    """
    Extracts and processes data blocks from the PSRFITS data.
    """
    DATA = data_all["DATA"]
    DAT_SCL = data_all["DAT_SCL"]
    DAT_OFFS = data_all["DAT_OFFS"]
    DAT_FREQ = data_all["DAT_FREQ"]
    return DATA, DAT_SCL, DAT_OFFS, DAT_FREQ

def calculate_statistics(DATA, DAT_SCL, DAT_OFFS):
    """
    Calculates permutation entropy and correlation dimension for single pulses.
    """
    num_pulses = np.shape(DATA)[0]
    num_bins = np.shape(DATA)[3]
    
    perm_entropy_single_pulses = np.zeros([num_pulses, 1])
    corr_dims_single_pulses = np.zeros([num_pulses, 1])
    
    for i in range(num_pulses):
        Single_Pulses = DATA[i, 0, 0, :] * DAT_SCL[i] + DAT_OFFS[i]
        # Calculate permutation entropy
        perm_entropy_single_pulses[i] = ant.perm_entropy(Single_Pulses, delay=[1, 2, 3, 4], order=10, normalize=True)
        # Calculate fractal dimension
        corr_dims_single_pulses[i] = nolds.corr_dim(Single_Pulses, emb_dim=2)
    
    return perm_entropy_single_pulses, corr_dims_single_pulses

def plot_histogram(values, title, output_path, xlabel='Value', ylabel='Probability Density'):
    """
    Plots a histogram of the given values and saves the figure.
    """
    plt.hist(values, density=True, alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=400)
    plt.show()

def calculate_average_pulse(DATA, DAT_SCL, DAT_OFFS):
    """
    Calculates the average pulse profile.
    """
    num_bins = np.shape(DATA)[3]
    Flux_all = np.zeros([num_bins, 1])
    for i in range(num_bins):
        Flux_all[i] = np.sum(DATA[:, 0, 0, i] * DAT_SCL[:] + DAT_OFFS[:])
    return Flux_all

def plot_average_pulse(Flux_all, output_path):
    """
    Plots the average pulse profile and saves the figure.
    """
    plt.plot(Flux_all / np.max(Flux_all))
    plt.xlabel('Phase')
    plt.ylabel('Normalized Total Flux')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=400)
    plt.close()

def process_file(file_path, output_dir):
    """
    Processes a single PSRFITS file and generates statistics and plots.
    """
    psrfit_data, pulsar_name, data_all = read_psrfits_data(file_path)
    DATA, DAT_SCL, DAT_OFFS, DAT_FREQ = process_data_block(data_all)
    
    # Calculate statistics
    perm_entropy_single_pulses, corr_dims_single_pulses = calculate_statistics(DATA, DAT_SCL, DAT_OFFS)
    
    # Plot histogram of permutation entropy
    histogram_path = os.path.join(output_dir, 'perm_entropy-delay-1-4-order-10', pulsar_name + '.png')
    os.makedirs(os.path.dirname(histogram_path), exist_ok=True)
    plot_histogram(perm_entropy_single_pulses, pulsar_name, histogram_path, xlabel='Permutation Entropy')

    # Plot histogram of correlation dimension
    histogram_path = os.path.join(output_dir, 'corr_dim-emb_dim-2', pulsar_name + '.png')
    os.makedirs(os.path.dirname(histogram_path), exist_ok=True)
    plot_histogram(corr_dims_single_pulses, pulsar_name, histogram_path, xlabel='Correlation Dimension')

    # Calculate and plot average pulse profile
    Flux_all = calculate_average_pulse(DATA, DAT_SCL, DAT_OFFS)
    pulse_profile_path = os.path.join(output_dir, 'PulseProfiles', pulsar_name + '.png')
    plot_average_pulse(Flux_all, pulse_profile_path)

def main():
    """
    Main function to process all files in parallel.
    """
    file_names = [
        'f210820_041554.DFp', 
        'f210820_105959.DFp',
        'f210823_120959.DFp',
        'f210823_124534.DFp',
        'f210828_190858.DFp',
        'f210906_152306.DFp',
        'f210920_142859.DFp',
        'f210921_002549.DFp',
        'f210921_004724.DFp',
        'f210922_085559.DFp',
        'f210929_090945.DFp'
    ]
    
    input_dir = 'Wang/single_pulse_DFp/'
    output_dir = 'statFigs/'
    
    # Get the number of available CPUs
num_cpus = os.cpu_count()

# Use a subset of CPUs (e.g., half of the available CPUs)
with ProcessPoolExecutor(max_workers=num_cpus // 2) as executor:
    futures = []
    for address in file_names:
        file_path = os.path.join(input_dir, address)
        futures.append(executor.submit(process_file, file_path, output_dir))
    
    # Wait for all tasks to complete
    for future in futures:
        future.result()

if __name__ == "__main__":
    main()