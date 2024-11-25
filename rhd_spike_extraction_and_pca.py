import datetime
import sys
import time

import matplotlib.pyplot as plt

import numpy as np

import os

from intanutil.header import (read_header,
                              header_to_result)
from intanutil.data import (calculate_data_size,
                            read_all_data_blocks,
                            check_end_of_file,
                            parse_data,
                            data_to_result)

from intanutil.filter import apply_notch_and_highpass_filter

from scipy.signal import find_peaks
from scipy.stats import mode
from sklearn.decomposition import PCA

def read_data(filename):
    # Start measuring how long this read takes
    tic = time.time()

    #Open the file for reading
    with open(filename, 'rb') as fid:

        #Read header and summarize its contents o console
        header = read_header(fid)

        # Calculate how much data is present and summarize to console.
        data_present, filesize, num_blocks, num_samples = (
            calculate_data_size(header, filename, fid))

        # If .rhd file contains data, read all present data blocks into 'data'
        # dict, and verify the amount of data read.
        if data_present:
            data = read_all_data_blocks(header, num_samples, num_blocks, fid)
            check_end_of_file(filesize, fid)

    # Save information in 'header' to 'result' dict.
    result = {}
    header_to_result(header, result)

    # If .rhd file contains data, parse data into readable forms and, if
    # necessary, apply the same notch filter that was active during recording.
    if data_present:
        parse_data(header, data)
        apply_notch_and_highpass_filter(header, data)

        # Save recorded data in 'data' to 'result' dict.
        data_to_result(header, data, result)

    # Otherwise (.rhd file is just a header for One File Per Signal Type or
    # One File Per Channel data formats, in which actual data is saved in
    # separate .dat files), just return data as an empty list.
    else:
        data = []

    # Report how long read took.
    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))

    # Return 'result' dict.
    return result

def auto_determine_classify_interval(spike_times, method, percentile):
    #Calculate Inter-Spike Intervals
    isi = np.diff(spike_times) * 1000 #Convert ISI to milliseconds
    print(f"ISI values (ms): {isi}")

    if method == 'mode':
        # Use mode
        mode_value = mode(isi)[0][0]
        classify_interval_ms = mode_value
    elif method == 'percentile':
        #Use percentile
        classify_interval_ms = np.percentile(isi, percentile)
    elif method == 'median':
        #use the median
        classify_interval_ms = np.median(isi)
    else:
        raise ValueError("Invalid method.  Use 'mode', 'percentile', or 'median'.")

    print(f"Automatically determined classify interval: {classify_interval_ms:.2f} ms (Method: {method})")
    return classify_interval_ms

def extract_and_classify_spikes(result, channel_index, threshold_ratio, pre_time, post_time, n_components, std_method, std_param):
    
    if 'amplifier_data' not in result or 't_amplifier' not in result:
        print("Amplifier data not found in the result dictionary.")
        return

    # Extract the signal and time
    signal = result['amplifier_data'][channel_index]
    time = result['t_amplifier']
    sample_rate = result['sample_rate']
    if sample_rate is None:
        raise ValueError("Sample rate not found in result. Ensure the header information is correctly saved.")
    
    # Convert pre_time and post_time to sample counts
    pre_samples = int((pre_time / 1000) * sample_rate)  # ms to samples.
    post_samples = int((post_time / 1000) * sample_rate)

    #convert signal so that it can be used for calculation

    # Dynamically calculate the threshold based on the maximum amplitude
    max_amplitude = np.max(signal)
    threshold = float(threshold_ratio) * max_amplitude
    print(f"Using dynamic threshold: {threshold} μV (Ratio: {threshold_ratio}, Max amplitude: {max_amplitude} μV)")

    #define the threshold range
    #min_height = (1/3) * max_amplitude
    #max_height = (1/2) * max_amplitude

    # Detect spikes based on the dynamic threshold range
    #spikes, _ = find_peaks(signal, height=(min_height, max_height))
    #print(f"Detected {len(spikes)} spikes within the range {min_height:.2f} to {max_height:.2f} μV.")

    spikes, _ = find_peaks(signal, height=threshold)
    print(f"Detected {len(spikes)} spikes with the threshold {threshold} μV.")
    # Extract spike data
    spike_data = []
    aligned_time = np.linspace(-pre_time, post_time, pre_samples + post_samples)
    std_values = []
    for spike in spikes:
        start = max(0, spike - pre_samples)
        end = min(len(signal), spike + post_samples)
        if end - start == (pre_samples + post_samples):  # Ensure all windows are the same size
            spike_waveform = signal[start:end]
            spike_time = time[spike]
            spike_data.append({
                'time': spike_time,
                'waveform': spike_waveform
            })
            std_values.append(np.std(spike_waveform))

    #Dynamically calculate std_threshold
    if std_method == 'mean':
        std_threshold = np.mean(std_values)
    elif std_method == 'median':
        std_threshold = np.median(std_values)
    elif std_method == 'percentile':
        std_threshold = np.percentile(std_values, std_param)
    else:
        raise ValueError("Invalid std_method. Choose 'mean', 'median', or 'percentile'.")

    print(f"Dynamically calculated std_threshold: {std_threshold:.2f} (Method: {std_method}, Param: {std_param})")

    #Convert classify_interval_ms to seconds for comparison
    spike_times = np.array([s['time'] for s in spike_data])
    classify_interval_ms = auto_determine_classify_interval(spike_times, 'percentile', 10)
    classify_interval_sec = classify_interval_ms / 1000

    #Classify spikes based on intervals
    isi = np.diff(spike_times)
    regular_spikes = []
    random_spikes = []

    #for i in range(len(isi)):
        #if isi[i] <= classify_interval_sec:
            #regular_spikes.append(i)
            #regular_spikes.append(i + 1)
        #else:
            #random_spikes.append(i + 1)

    #Calculate the standard deviation and choose those are above the threshold
    for i, spike in enumerate(spike_data):
        std_dev = np.std(spike['waveform'])
        if std_dev >= std_threshold:
            regular_spikes.append(i)

    print(f"Filtered {len(regular_spikes)} regular spikes based on standard deviation threshold: {std_threshold:.2f}")

    #Make sure there is no duplicates
    regular_spikes = np.unique(regular_spikes)
    #random_spikes = np.unique([i for i in range(len(spike_data)) if i not in regular_spikes])

    # Apply PCA to the spike data
    #pca = PCA(n_components=n_components)
    #spike_features = pca.fit_transform(spike_data)
    #print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_}")

    return signal, time, aligned_time, spike_data, spikes, regular_spikes, random_spikes

def save_first_n_spikes_as_png(spike_data, regular_spikes, n, save_dir):

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Select the first n regular spikes
    selected_spikes = [spike_data[i] for i in regular_spikes[:n]]
    
    # Save each spike as a PNG
    for idx, spike in enumerate(selected_spikes):
        plt.figure(figsize=(8, 4))
        waveform = spike['waveform']
        time = np.arange(len(waveform))  # Sample indices as time
        plt.plot(time, waveform, label=f"Spike {idx + 1} at {spike['time']:.3f}s")
        plt.axvline(x=len(waveform)//2, color='red', linestyle='--', label='Spike Time (t=0)')
        plt.title(f"Spike {idx + 1} Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude (μV)")
        plt.legend()
        
        # Save the figure as PNG
        save_path = os.path.join(save_dir, f"spike_{idx + 1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Spike {idx + 1} as {save_path}")



if __name__ == '__main__':
    filename = sys.argv[1]
    result = read_data(filename)
    signal, time, aligned_time, spike_data, spikes, regular_spikes, random_spikes= extract_and_classify_spikes(
        result, 
        0, 
        0.2, 
        1500, 
        2500, 
        2, 
        'percentile', 
        90
        )


    if signal is not None:
        #Save the current time
        import datetime
        currrent_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        #Choose where to save the directory
        save_dir = "/home/u634567g/for_brainwave/spike_figures"

        #Save the first 10 spikes
        save_first_n_spikes_as_png(
        spike_data,
        regular_spikes,
        30,
        "spike_images")

        # Plot spike waveforms
        #plt.figure(figsize=(12, 6))
        #for waveform in spike_waveforms:
            #plt.plot(aligned_time, waveform, alpha=0.5, color='gray')
        #plt.axvline(x=0, color='red', linestyle='--', label='Spike Time (t=0)')
        #plt.title("Aligned Spike Waveforms (Channel 0)")
        #plt.xlabel("Time (ms)")
        #plt.ylabel("Amplitude (μV)")
        #plt.legend()
        #custom_waveform_name = input("Name a graph for spike waveform (the file will be saved as: the_name_you_entered_spike_waveform_current_time.png): ")
        #waveform_filename = f"{custom_waveform_name}_spike_waveform_{currrent_time}.png"
        #plt.savefig(waveform_filename)
        #print(f"graph is now saved: {waveform_filename}")

        #Plot signal and classified spikes on the graph
        #plt.figure(figsize=(12, 6))
        #plt.plot(time, signal, label='Signal', color='black', alpha=0.7)

        #Plot regular spikes
        #for i in regular_spikes:
            #spike_time = spike_data[i]['time']
            #spike_waveform = spike_data[i]['waveform']
            #plt.scatter(spike_time, np.max(spike_waveform), color='blue', label='Regular Spike' if i == regular_spikes[0] else "")

        # Plot random spikes
        #for i in random_spikes:
            #spike_time = spike_data[i]['time']
            #spike_waveform = spike_data[i]['waveform']
            #plt.scatter(spike_time, np.max(spike_waveform), color='orange', label='Random Spike' if i == random_spikes[0] else "")

        #plt.title("Signal with Classified Spikes")
        #plt.xlabel("Time (s)")
        #plt.ylabel("Amplitude (μV)")
        #plt.legend()
        #custom_classified_spikes_name = input("Name the graph for classified spikes (the file will be saved as: the_name_you_entered_classified_spikes_current_time.png): ")
        #classified_spikes_filename = f"{custom_classified_spikes_name}_classified_spikes_{currrent_time}.png"
        #plt.savefig(classified_spikes_filename)
        #print(f"Graph is now saved: {classified_spikes_filename}")

        #Visualize each regular spike's waveform
        #for i in regular_spikes:
            #spike_time = spike_data[i]['time']
            #spike_waveform = spike_data[i]['waveform']
            #plt.figure(figsize=(8, 4))
            #plt.plot(aligned_time, spike_waveform, label=f"Spike at {spike_time:.3f}s")
            #plt.axvline(x=0, color='red', linestyle='--', label='Spike Time (t=0)')
            #plt.title(f"Spike Waveform (Standard Deviation: {np.std(spike_waveform):.2f})")
            #plt.xlabel("Time (ms)")
            #plt.ylabel("Amplitude (μV)")
            #plt.legend()
            #plt.savefig("demo")
            #plt.close()

        # Plot original signal with spikes highlighted
        plt.figure(figsize=(60, 6))
        plt.plot(time, signal, label='Amplitude (Signal)')
        plt.scatter(time[spikes], signal[spikes], color='red', label='Detected Spikes')
        plt.title("Amplitude with Detected Spikes (Channel 0)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (μV)")
        plt.legend()
        custom_highlighted_spikes_name = input("Name a graph for highlighted spikes (the file will be saved as: the_name_you_entered_highlighted_spikes_current_time.png): ")
        highlighted_spikes_filename = os.path.join(save_dir, f"{custom_highlighted_spikes_name}_highlighted_spikes_{currrent_time}.png")
        plt.savefig(highlighted_spikes_filename)
        print(f"graph is now saved: {highlighted_spikes_filename}")
        plt.close()


        #for i in regular_spikes:
            #spike_time = spike_data[i]['time']
            #spike_waveform = spike_data[i]['waveform']

            #spike_waveforms = np.array(spike_waveform)
            #for waveform in spike_waveforms:
                #plt.plot(waveform, alpha=0.5, color='gray')
            #plt.title(f"Spike Waveforms")
            #plt.xlabel("Time (samples)")
            #plt.ylabel("Amplitude (μV)")
            #plt.legend()
            #plt.show()
            #plt.savefig("demo_regular_spikes")
            #plt.close()

        # Plot PCA results
        #if pca.n_components == 2:
            #plt.figure(figsize=(10, 6))
            #plt.scatter(spike_features[:, 0], spike_features[:, 1], alpha=0.7, color='blue')
            #plt.title("PCA of Spike Waveforms (2D Projection)")
            #plt.xlabel("Principal Component 1")
            #plt.ylabel("Principal Component 2")
            #custom_pca_name = input("Name a graph for pca (the file will be saved as: the_name_you_entered_pca_current_time.png): ")
            #pca_filename = f"{custom_pca_name}_pca_{currrent_time}.png"
            #plt.savefig(pca_filename)
            #print(f"graph is now saved: {pca_filename}")
            #plt.close()
        #elif pca.n_components == 3:
            #from mpl_toolkits.mplot3d import Axes3D
            #fig = plt.figure(figsize=(10, 8))
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(spike_features[:, 0], spike_features[:, 1], spike_features[:, 2], alpha=0.7, color='blue')
            #ax.set_title("PCA of Spike Waveforms (3D Projection)")
            #ax.set_xlabel("Principal Component 1")
            #ax.set_ylabel("Principal Component 2")
            #ax.set_zlabel("Principal Component 3")
            #custom_pca_name = input("Name a graph for pca (the file will be saved as: the_name_you_entered_pca_current_time.png): ")
            #pca_filename = f"{custom_pca_name}_pca_{currrent_time}.png"
            #plt.savefig(pca_filename)
            #print(f"graph is now saved: {pca_filename}")
            #plt.close()

        #a = read_data(sys.argv[1])
        #print(a)

        #fig, ax = plt.subplots(2, 1)
        #ax[0].set_ylabel('Amp')
        #ax[0].plot(a['t_amplifier'], a['amplifier_data'][0, :])
        #ax[0].margins(x=0, y=0)

        #ax[1].set_ylabel('Aux')
        #ax[1].plot(a['t_aux_input'], a['aux_input_data'][2, :])
        #ax[1].margins(x=0, y=0)

    # ax[2].set_ylabel('Vdd')
    # ax[2].plot(a['t_supply_voltage'], a['supply_voltage_data'][0, :])
    # ax[2].margins(x=0, y=0)

    # ax[3].set_ylabel('ADC')
    # ax[3].plot(a['t_board_adc'], a['board_adc_data'][0, :])
    # ax[3].margins(x=0, y=0)

    # ax[4].set_ylabel('Digin')
    # ax[4].plot(a['t_dig'], a['board_dig_in_data'][0, :])
    # ax[4].margins(x=0, y=0)

    # ax[5].set_ylabel('Digout')
    # ax[5].plot(a['t_dig'], a['board_dig_out_data'][0, :])
    # ax[5].margins(x=0, y=0)

    # ax[6].set_ylabel('Temp')
    # ax[6].plot(a['t_temp_sensor'], a['temp_sensor_data'][0, :])
    # ax[6].margins(x=0, y=0)

        #custom_name = input("Name a graph for the original brain signal graph (the file will be saved as: the_name_you_entered_original_brain_signal_graph_current_time.png): ")
        #filename = f"{custom_name}_original_graph_{currrent_time}.png"
        #plt.savefig(filename)
        #print(f"graph is now saved: {filename}")
        #plt.close()
