import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from networkx import goldberg_radzik
import pandas as pd

def get_wct(results:pd.DataFrame) -> int:
    """get whole completion time.
    
    Args: 
        results dataframe of the workflow
    Returns:
        workflow completion time from task created to task results received.
    """
    start_time = results['time_created'].min()
    end_time = results['time_result_received'].max()
    return int(end_time - start_time)

def time_line_graph(results: pd.DataFrame, task_color: dict):
    """plot time_line_graph of each task.
    use at  plot workflow analysis.ipynb

    Args:
        results (pd.DataFrame): results object of the workflow from colmema.
        task_color (dict): color of each task.  
    """ 

    fig,ax = plt.subplots(figsize=(20, 12))
    task_timeline = results
    task_timeline.sort_values('time_compute_started', inplace=True, ignore_index=True)
    start_time = task_timeline['time_compute_started'].loc[0]
    wct = get_wct(task_timeline)
    ax.text(0, len(task_timeline)-1, f'  Workflow completion time: {wct}', ha='left', va='center', fontsize=30, color='red')
    handles = []
    labels = []
    for index, row in task_timeline.iterrows():
        print(f"{row['method']}: {row['time_compute_started']}: {row['time_running']}")
        # method = str(row['method'])+str(index)+"  time:  "+str(row['time_running'])
        method = str()
        time = row['time_running']
        bar = ax.barh(index, time, left=row['time_compute_started'] - start_time, color=task_color[row['method']])
        # ax.text(row['time_compute_started'] - start_time, index, method, ha='left', va='center',fontsize=6)
    
    ax.set_xlabel('Time (s)',fontsize=34)
    ax.set_ylabel('Task',fontsize=34)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    handles = []
    labels = []
    for task, color in task_color.items():
        patch = mpatches.Patch(color=color, label=task)
        handles.append(patch)
        labels.append(task)
    ax.legend(handles, labels, loc='lower right', prop={'size': 30})
    # ax.margins(y=0.1)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.grid()
    plt.show()
    

## plt.scatter
def plot_scatter(x, y, x_label, y_label, title, save_path):
    """plot scatter graph.

    Args:
        x (list): x axis data.
        y (list): y axis data.
        x_label (str): x axis label.
        y_label (str): y axis label.
        title (str): title of the graph.
        save_path (str): save path of the graph.
    """    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(x, y, s=100, alpha=0.5)
    ax.set_xlabel(x_label,fontsize=34)
    ax.set_ylabel(y_label,fontsize=34)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(title, fontsize=34)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def get_gpu_data(gpu_log:str) -> pd.DataFrame:
    df = pd.read_csv(gpu_log)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    gpu_utilization_group = df.groupby('GPUs')['GPU Utilization (%)']
    gpu = [ gpu_utilization_group.get_group(i) for i in range(0, len(gpu_utilization_group.groups.keys()))]
    return gpu

def plot_gpu_util(gpu:list[pd.DataFrame]=None) -> None:
    if gpu is None:
        gpu = get_gpu_data(gpu_log)
    # plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange']
    # for i in range(len(gpu)):
    for i in range(0,1):
        # plt.plot([i for i in range(len(gpu[i]))],gpu[i], label=f'GPU {i}', color=colors[i], linestyle='--')
        plt.plot([i for i in range(len(gpu[i][0:1000]))],gpu[i][0:1000], label=f'GPU', color=colors[i], alpha=0.5)
        plt.legend(loc='upper right', fontsize=18)
    # plt.plot(gpu[0], label='GPU 0', color='red', linestyle='--')
    plt.xlabel('Timestamp (s)',fontsize=20)
    plt.ylabel('GPU Utilization (%)',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax = plt.gca()  # 获取当前坐标轴对象
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # 设置x轴刻度的最大数量为5个
    # plt.title('GPU Utilization Over Time')
    plt.grid(True)
    
def simple_bar_plot(idx, values, left, color, save_path=None):
    """plot simple bar graph.

    """
    
    fig, ax = plt.subplots(figsize=(20, 12))
    for i in range(len(idx)):
        ax.bar(idx[i], values[i], left=left[i], color=color[i])
    ax.bar(idx, values, left=left, color=color)
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)