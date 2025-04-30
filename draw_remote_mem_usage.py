import matplotlib.pyplot as plt

colors = ['#d20962', '#7ac143']
plt.rcParams['font.size'] = 14

def read_data(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]
    return data

def plot_data(file1, file2, label1, label2):

    data1 = read_data(file1)
    data2 = read_data(file2)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(data1, label=label1, linestyle='-', linewidth=3, color=colors[0])
    plt.plot(data2, label=label2, linestyle='-', linewidth=3, color=colors[1])
    
    plt.xlabel('Time(s)')
    plt.ylabel('Remote Memory(GB)')
    
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("figures/remote_mem_usage.png")
    plt.savefig("figures/remote_mem_usage.pdf")

if __name__ == "__main__":
    file1 = 'remote_mem_usage/finememswap.txt'  
    file2 = 'remote_mem_usage/fastswap.txt' 
    
    plot_data(file1, file2, 
              label1='FineMem-Swap', 
              label2='FastSwap')