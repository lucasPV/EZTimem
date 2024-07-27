#!/usr/bin/env python3

import sys
import psutil
import time
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from datetime import datetime
from typing import Dict, List
from prettytable import PrettyTable


def monitor_resources(pid: int, monitor_interval: float, log: Dict[str, List]):
    """
    Monitors and logs various system and process-related resources for a given process over time.
    It tracks CPU usage, memory usage, disk I/O, context switches, and if available, GPU values.
    Data is collected at specified intervals.

    Parameters:
    - pid (int): Process ID of the process to monitor.
    - monitor_interval (float): Time in seconds between data collection points.
    - log (Dict[str, List]): Dictionary where collected data will be appended.
    """
    try:
        process = psutil.Process(pid)
        start_time = time.time()
        # Initialize CPU usage percentage tracking
        process.cpu_percent(interval=None)
        total_memory = psutil.virtual_memory().total
        cpu_count = psutil.cpu_count()
        
        while process.is_running():
            # Gather resource usage data
            with process.oneshot():
                mem_info = process.memory_info()
                cpu_usage = process.cpu_percent(interval=None) / cpu_count
                io_counters = process.io_counters()
                memory_percent = (mem_info.rss / total_memory) * 100
                context_switches = process.num_ctx_switches()
                process_priority = process.nice()

                # Gather GPU data
                gpus = GPUtil.getGPUs()
                gpu_load = [gpu.load for gpu in gpus]
                # GPU memory will be kept in bytes for consistency
                gpu_memory_used = [gpu.memoryUsed*(1024**2) for gpu in gpus]
                gpu_memory_total = [gpu.memoryTotal*(1024**2) for gpu in gpus]
                gpu_memory_util = [(gpu.memoryUsed / gpu.memoryTotal) * 100
                                   if gpu.memoryTotal else 0 for gpu in gpus]
                
                # Appending new values
                log["memory_rss"].append(mem_info.rss)
                log["memory_vms"].append(mem_info.vms)
                log["cpu_usage"].append(cpu_usage)
                log["io_read_bytes"].append(io_counters.read_bytes)
                log["io_write_bytes"].append(io_counters.write_bytes)
                log["memory_usage_percent"].append(memory_percent)
                log["threads"].append(process.num_threads())
                log["open_files"].append(len(process.open_files()))
                log["voluntary_ctxt_switches"].append(context_switches.voluntary)
                log["involuntary_ctxt_switches"].append(context_switches.involuntary)
                log["process_priority"].append(process_priority)
                log["gpu_load"].append(gpu_load)
                log["gpu_memory_used"].append(gpu_memory_used)
                log["gpu_memory_util"].append(gpu_memory_util)

            # Update track of time
            current_time = time.time() - start_time
            log["times"].append(current_time)

            # Wait for interval
            time.sleep(monitor_interval)
    except Exception as e:
        pass


def init_empty_log() -> Dict[str, List]:
    """
    Initialize and return an empty log dictionary.

    Returns:
        Dict[str, List]: A dictionary with keys representing different system metrics,
                         each associated with an empty list to collect data.
    """
    return {
        "times": [],
        "memory_rss": [],
        "memory_vms": [],
        "cpu_usage": [],
        "io_read_bytes": [],
        "io_write_bytes": [],
        "memory_usage_percent": [],
        "threads": [],
        "open_files": [],
        "voluntary_ctxt_switches": [],
        "involuntary_ctxt_switches": [],
        "process_priority": [],
        "gpu_load": [],
        "gpu_memory_used": [],
        "gpu_memory_util": []
    }


def transpose_list(matrix):
    """
    Transpose a given 2D list (matrix).

    This function takes a two-dimensional list (matrix) and returns a new matrix which is the transpose
    of the input. The transpose of a matrix is formed by turning rows into columns and vice versa.

    Args:
        matrix (List[List[Any]]): A 2D list representing the original matrix to be transposed.

    Returns:
        List[List[Any]]: A new 2D list representing the transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]


def memory_to_str(memory_value):
    """
    Converts a memory size from bytes to a human-readable string using
    megabytes (MB) or gigabytes (GB) as units.
    
    Parameters:
    - memory_value (int or float): Memory size in bytes.
    
    Returns:
    - str: A string representing the memory size in MB or GB with two decimal places.
    """
    # Check if the memory value is at least 1 gigabyte
    if memory_value >= 1000**3:  # 1000^3 bytes = around 1 gigabyte
        # Convert from bytes to gigabytes and format to two decimal places
        return f"{memory_value / (1024**3):.2f} GB"
    else:
        # Convert from bytes to megabytes and format to two decimal places
        return f"{memory_value / (1024**2):.2f} MB"


def display_statistics(log: Dict[str, List]):
    """
    Displays a summary table and printouts of reports considering the resources used
    by the command (application) that was monitored.

    Parameters:
    - log (Dict[str, List]): A dictionary containing lists of system process statistics. 
    """
    # Initialize a pretty table for displaying process statistics
    table = PrettyTable()
    table.align = "l"  # Left align text in cells
    table.field_names = ["Process Report", "Average", "Std. Dev.", "Max."]  # Set column headers

    # Calculate total disk read and write by subtracting the first from the last recorded value
    read_bytes = log["io_read_bytes"][-1] - log["io_read_bytes"][0]
    write_bytes = log["io_write_bytes"][-1] - log["io_write_bytes"][0]
    # Print total disk I/O
    print(" Total Disk Read:", memory_to_str(read_bytes))
    print(" Total Disk Write:", memory_to_str(write_bytes))
    # Print total context switches
    print(" Total Voluntary Ctxt Switch:", log["voluntary_ctxt_switches"][-1])
    print(" Total Involuntary Ctxt Switch:", log["involuntary_ctxt_switches"][-1])

    # Mode identifiers for formatting output
    mode_default = 0
    mode_percent = 1
    mode_bytes = 2

    # Dictionary mapping keys to labels and modes for table formatting
    table_keys = {
        "cpu_usage": ["CPU Usage", mode_percent],
        "memory_usage_percent": ["Memory (%)", mode_percent],
        "memory_rss": ["Memory (RSS)", mode_bytes],
        "memory_vms": ["Memory (VMS)", mode_bytes],
        "threads": ["Threads Used", mode_default],
        "open_files": ["Open Files", mode_default],
        "process_priority": ["Process Priority", mode_default],
    }

    # Process each metric and add rows to the table
    for key, info in table_keys.items():
        label, print_mode = info
        values = log[key]
        if values:  # Only process non-empty lists
            if isinstance(values[0], list):  # Handle nested lists (e.g., per-thread values)
                values = [item for sublist in values for item in sublist]
            avg = np.mean(values)  # Compute average
            std = np.std(values)   # Compute standard deviation
            max_value = np.max(values)  # Compute maximum value

            # Add row to table based on the specified print mode
            if print_mode == mode_default:
                table.add_row([label, f"{avg:.2f}", f"{std:.2f}", f"{max_value:.2f}"])
            elif print_mode == mode_percent:
                table.add_row([label, f"{avg:.2f} %", f"{std:.2f} %", f"{max_value:.2f} %"])
            elif print_mode == mode_bytes:
                table.add_row([label, memory_to_str(avg), memory_to_str(std), memory_to_str(max_value)])

    # Print the formatted table
    print(table)


def display_statistics_gpu(log: Dict[str, List]):
    """
    Displays a summary table of GPU statistics including average, standard deviation, and maximum values
    for GPU load, memory usage, and memory utilization.
    Data is reported for all GPUs.

    Parameters:
    - log (Dict[str, List]): A dictionary containing lists of GPU statistics. Each key in the dictionary
      ('gpu_load', 'gpu_memory_used', 'gpu_memory_util') must contain a list of lists where each sub-list
      represents the values for a respective GPU across different times.
    """
    # Initialize a pretty table for displaying GPU statistics
    table_gpu = PrettyTable()
    table_gpu.align = "l"  # Left align text in cells
    table_gpu.field_names = ["GPU Report", "Average", "Std. Dev.", "Max."]  # Set column headers

    # Number of GPUs based on the length of first list in gpu_load
    num_gpus = len(log["gpu_load"][0])

    # Transposing lists for easier statistics computation
    log["gpu_load"] = transpose_list(log["gpu_load"])
    log["gpu_memory_used"] = transpose_list(log["gpu_memory_used"])
    log["gpu_memory_util"] = transpose_list(log["gpu_memory_util"])
    
    # Mode identifiers for how to print the statistics
    mode_default = 0
    mode_percent = 1
    mode_bytes = 2

    # Dictionary to map log keys to labels and modes
    table_gpu_keys = {
        "gpu_load": ["Usage", mode_percent],
        "gpu_memory_used": ["Memory", mode_bytes],
        "gpu_memory_util": ["Memory (%)", mode_percent],
    }

    # Iterate through each GPU and each type of statistic
    for gpu_count in range(num_gpus):
        for key, info in table_gpu_keys.items():
            label, print_mode = info
            values = log[key][gpu_count]  # Extract data for current GPU
            avg = np.mean(values)         # Compute average
            std = np.std(values)          # Compute standard deviation
            max_value = np.max(values)    # Compute maximum value

            # Format label depending on the number of GPUs
            if num_gpus <= 1:
                label = "GPU {}".format(label)
            else:
                label = "GPU {} {}".format(gpu_count, label)

            # Add row to table based on the mode specified
            if print_mode == mode_default:
                table_gpu.add_row([label, f"{avg:.2f}", f"{std:.2f}", f"{max_value:.2f}"])
            elif print_mode == mode_percent:
                table_gpu.add_row([label, f"{avg:.2f} %", f"{std:.2f} %", f"{max_value:.2f} %"])
            elif print_mode == mode_bytes:
                table_gpu.add_row([label, memory_to_str(avg), memory_to_str(std), memory_to_str(max_value)])

    # Print the table
    print(table_gpu)


def plot_resources(log: Dict[str, List]):
    """
    Plots the CPU usage, number of threads, RSS memory, VMS memory, and number of open files over time
    and saves the plot to a file named 'eztimem_plots.png', with legends placed outside the graph on the right side.

    Parameters:
    - log (Dict[str, List]): Dictionary containing the logged data for plotting.
    """
    # Convert times from seconds to minutes for a more readable format
    times_in_minutes = [t / 60 for t in log['times']]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 24))  # Adjusted subplot count and figure size

    # Get correct length
    correct_len = min(len(log['cpu_usage']), len(times_in_minutes))

    # Plot CPU usage
    axs[0].plot(times_in_minutes, log['cpu_usage'][:correct_len], label='CPU Usage (%)', color='blue')
    axs[0].set_title('CPU Usage Over Time')
    axs[0].set_xlabel('Time (minutes)')
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].grid(True)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot Number of Threads
    axs[1].plot(times_in_minutes, log['threads'][:correct_len], label='Number of Threads', color='purple')
    axs[1].set_title('Number of Threads Over Time')
    axs[1].set_xlabel('Time (minutes)')
    axs[1].set_ylabel('Number of Threads')
    axs[1].grid(True)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot RSS Memory
    rss_memory_in_mb = [m / (1024 ** 2) for m in log['memory_rss']]  # Convert from bytes to MB
    axs[2].plot(times_in_minutes, rss_memory_in_mb[:correct_len], label='Memory RSS (MB)', color='red')
    axs[2].set_title('RSS Memory Usage Over Time')
    axs[2].set_xlabel('Time (minutes)')
    axs[2].set_ylabel('Memory RSS (MB)')
    axs[2].grid(True)
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot VMS Memory
    vms_memory_in_mb = [m / (1024 ** 2) for m in log['memory_vms']]  # Convert from bytes to MB
    axs[3].plot(times_in_minutes, vms_memory_in_mb[:correct_len], label='Memory VMS (MB)', color='green')
    axs[3].set_title('VMS Memory Usage Over Time')
    axs[3].set_xlabel('Time (minutes)')
    axs[3].set_ylabel('Memory VMS (MB)')
    axs[3].grid(True)
    axs[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot Number of Open Files
    axs[4].plot(times_in_minutes, log['open_files'][:correct_len], label='Open Files', color='orange')
    axs[4].set_title('Open Files Over Time')
    axs[4].set_xlabel('Time (minutes)')
    axs[4].set_ylabel('Open Files')
    axs[4].grid(True)
    axs[4].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Tight layout before saving to handle overlaps
    plt.tight_layout()
    # Save the figure
    export_file = 'eztimem_plots.png'
    plt.savefig(export_file, bbox_inches='tight')  # Adjust saving to include external legend
    plt.close(fig)  # Close the plot figure to free memory
    # Print to user
    print("----------------------------------------")
    print(" Plots saved in {}".format(export_file))
    print("----------------------------------------")


if __name__ == "__main__":
    # Check if command was provided as input
    if len(sys.argv) <= 1:
        print("Call me like this: python {} <command>".format(sys.argv[0]))
        sys.exit(1)

    # Parse command
    command = " ".join(sys.argv[1:])

    # Dictionary to store the report data
    log = init_empty_log()

    # Interval to monitor process (in seconds)
    monitor_interval = 0.1

    # Run process and monitor it
    date_start = datetime.now()  # get start datetime
    start_time = time.perf_counter()  # start timer
    process = psutil.Popen(command.split())  # start process
    monitor_thread = Thread(target=monitor_resources, args=(process.pid,
                                                            monitor_interval,
                                                            log))  # monitor process
    monitor_thread.start()
    process.wait()
    monitor_thread.join()
    end_time = time.perf_counter()  # end timer
    date_end = datetime.now()  # get end datetime
    elapsed_time = end_time - start_time  # calculate elapsed time

    # Check if command actually run
    if (len(log["cpu_usage"]) == 1) and log["cpu_usage"][0] == 0:
        print("No monitoring was done... Aborting.")
        sys.exit(1)

    # Report measured values
    print("\n==========================================================")
    print("=                     EZTIMEM REPORT                     =")
    print("==========================================================")
    # Elapsed time
    print(f" Command: {command}")
    print(f" Start Time: {date_start}")
    print(f" Finish Time: {date_end}")
    print(f" Total Time: {elapsed_time:.2f} seconds")
    # Print tables
    display_statistics(log)
    display_statistics_gpu(log)
    # Plot some graphs
    plot_resources(log)
    # To help the user
    print(" For your information:")
    print("* EZTIMEM monitoring checks are made every {} seconds.".format(monitor_interval))
    print("* CPU usage considers all cores.")
    print("* VMS Memory = total amount of virtual memory used by a process,")
    print("  including all code, data, and shared libraries plus pages that have")
    print("  been swapped out.")
    print("* RSS Memory = portion of memory occupied by a process that is held")
    print("  in RAM, excluding swapped out pages.")
    print("* Process Priority = default is zero.")
    print("* All reports consider only the resources utilized by the process,")
    print("  except for the GPU, which is reported for the entire system usage.")
