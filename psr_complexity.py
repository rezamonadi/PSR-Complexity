import pdat
import os
import numpy as np
np.bool = bool
import antropy as ant
import matplotlib.pyplot as plt
import nolds
import pywt
from tqdm import tqdm as tqd
import psrchive
from concurrent.futures import ProcessPoolExecutor 

def read_psrfits_data(file_path):
    """
    Reads PSRFITS data from the given file path.
    """
    if file_path.endswith('.DFp'):
        psrfit_data = pdat.psrfits(file_path)
        data_all = psrfit_data[2].read()
        PSR_name = data_all[0][0].decode('utf-8').strip()
        pulsar_name = PSR_name.split()[1]  # Extract pulsar name
        del data_all
        data_all = psrfit_data[4].read()

    if file_path.endswith('.ar'):
        arc = psrchive.Archive_load(file_path)
        PSR_name = arc.get_source()
        data_all = arc.get_data()
        # needs to be figured out --> understand data structure here

        
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

def wavelet_fractal_dimension(time_series, wavelet='db1'):
    """
    Calculates the fractal dimension of a time series using wavelet transforms.

    Parameters:
        time_series (array-like): The input time series.
        wavelet (str): The wavelet type to use (default is 'db1', Daubechies wavelet).

    Returns:
        float: The estimated fractal dimension.
    """
    # Perform the wavelet decomposition
    coeffs = pywt.wavedec(time_series, wavelet)

    # Calculate the energy at each decomposition level
    energies = [np.sum(np.abs(c)**2) for c in coeffs]

    # Calculate the log-log relationship between scale and energy
    scales = np.array([2**i for i in range(len(energies))])
    log_scales = np.log(scales)
    log_energies = np.log(energies)

    # Perform linear regression to estimate the slope
    slope, _ = np.polyfit(log_scales, log_energies, 1)

    # Fractal dimension is related to the slope
    fractal_dimension = -slope

    return fractal_dimension

def calculate_statistics(DATA, DAT_SCL, DAT_OFFS):
    """
    Calculates permutation entropy and correlation dimension for single pulses.
    """
    num_pulses = np.shape(DATA)[0]
    num_bins = np.shape(DATA)[3]
    
    perm_entropy_single_pulses = np.zeros([num_pulses, 1])
    corr_dims_single_pulses = np.zeros([num_pulses, 1])
    wavelet_fractal_dimension_single_pulses = np.zeros([num_pulses, 1])
    
    for i in range(num_pulses):
        Single_Pulses = DATA[i, 0, 0, :] * DAT_SCL[i] + DAT_OFFS[i]
        # Calculate permutation entropy
        perm_entropy_single_pulses[i] = ant.perm_entropy(Single_Pulses, delay=[1, 2, 3], order=10, normalize=True)
        
        # # Calculate fractal dimension (uncomment if needed)
        # corr_dims_single_pulses[i] = nolds.corr_dim(Single_Pulses, emb_dim=2)


        # Calculate fractal dimension using wavelet transform (uncomment if needed)
        wavelet_fractal_dimension_single_pulses[i] = wavelet_fractal_dimension(Single_Pulses)
    
    return perm_entropy_single_pulses, corr_dims_single_pulses, wavelet_fractal_dimension_single_pulses

def plot_histogram(values, title, output_path, bin_edges, xlabel='Value', ylabel='Probability Density'):
    """
    Plots a histogram of the given values and saves the figure.
    """
    plt.hist(values, density=True, alpha=0.75, bins=bin_edges)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=400)
    plt.close()

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
    
    print('Processing pulsar:', pulsar_name)
    # Calculate statistics
    perm_entropy_single_pulses, corr_dims_single_pulses, wavelet_fractal_dimension_single_pulses = calculate_statistics(DATA, DAT_SCL, DAT_OFFS)
    print('Plotting perm_entropy and corr_dims histograms')    
    
    # Plot histogram of permutation entropy
    histogram_dir = os.path.join(output_dir, 'permEntropy/perm_entropy-delay-1-3-order-10/')
    os.makedirs(os.path.dirname(histogram_dir), exist_ok=True)  # Ensure the directory exists
    histogram_path = os.path.join(histogram_dir, pulsar_name + '.png')
    plot_histogram(perm_entropy_single_pulses, pulsar_name, histogram_path, xlabel='Permutation Entropy', bins=np.linspace(0.9, 1, 100))

    # # Plot histogram of correlation dimension
    # histogram_dir = os.path.join(output_dir, 'fractal_dim/corr_dim-emb_dim-2/')
    # os.makedirs(os.path.dirname(histogram_dir), exist_ok=True)  # Ensure the directory exists
    # histogram_path = os.path.join(histogram_dir, pulsar_name + '.png')
    # plot_histogram(corr_dims_single_pulses, pulsar_name, histogram_path, xlabel='Correlation Dimension', bins=np.linspace(-0.1, 0.1, 100))

    # Plot histogram of wavelet fractal dimension
    histogram_dir = os.path.join(output_dir, 'wavelet_fractal_dim/db1/')
    os.makedirs(os.path.dirname(histogram_dir), exist_ok=True)  # Ensure the directory exists
    histogram_path = os.path.join(histogram_dir, pulsar_name + '.png')
    plot_histogram(wavelet_fractal_dimension_single_pulses, pulsar_name, histogram_path, xlabel='Wavelet Fractal Dimension', bins=np.linspace(-0.1, 0.1, 100))

    # # Calculate and plot average pulse profile
    # Flux_all = calculate_average_pulse(DATA, DAT_SCL, DAT_OFFS)
    # pulse_profile_path = os.path.join(output_dir, 'PulseProfiles', pulsar_name + '.png')
    # os.makedirs(os.path.dirname(pulse_profile_path), exist_ok=True)  # Ensure the directory exists
    # plot_average_pulse(Flux_all, pulse_profile_path)
def main():
    """
    Main function to process files.
    """

    # processing Wang Files
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
    output_dir = 'Wang/statFigs/'
    
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

    # # processing Vela Files
    # file_names = ['2021-05-10-070130.ar']
    
    
    # input_dir = 'Vela/single_pulses/'
    # output_dir = 'Vela/statFigs/'
    
    
    # # Use a subset of CPUs (e.g., half of the available CPUs)
    # with ProcessPoolExecutor(max_workers=num_cpus // 2) as executor:
 
    #     futures = []
    #     for address in file_names:
    #         file_path = os.path.join(input_dir, address)
    #         futures.append(executor.submit(process_file, file_path, output_dir))
        
    #     # Wait for all tasks to complete
    #     for future in futures:
    #         future.result()


if __name__ == "__main__":
    main()