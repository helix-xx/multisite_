import re
import configparser
import subprocess
import os

def parse_slurm_info(slurm_info):
    # 提取节点列表
    # node_list_match = re.search(r'\bNodeList=(\S+)', slurm_info)
    # node_list = node_list_match.group(1) if node_list_match else 'unknown'
    
    def get_nodes(node_list):
        # 解析节点列表
        nodes = []
        if '[' in node_list and ']' in node_list:
            prefix = node_list.split('[')[0]
            ranges = node_list.split('[')[1].split(']')[0].split(',')
            for r in ranges:
                if '-' in r:
                    start, end = map(int, r.split('-'))
                    for i in range(start, end + 1):
                        nodes.append(f"{prefix}{i:03}")
                else:
                    nodes.append(f"{prefix}{int(r):03}")
        else:
            nodes = node_list.split(',')
        return nodes

    # 提取每个节点的 CPU 和 GPU 信息
    node_info_pattern = re.compile(r'Nodes=(\S+) CPU_IDs=([\d,-]+) .* GRES=gpu:(\d+)')
    node_info = node_info_pattern.findall(slurm_info)
    
    # 计算 CPU 数量
    def count_cpus(cpu_ids):
        count = 0
        for part in cpu_ids.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                count += end - start + 1
            else:
                count += 1
        return count
    
    
    # 创建配置文件内容
    config_content = configparser.ConfigParser()
    # 添加Environment配置
    config_content['Environment'] = {
        'bashrc_path': os.path.expanduser('~/.bashrc'),
        'conda_path': os.path.expanduser('~/miniconda3'),
        'conda_env': os.path.expanduser('~/miniconda3/envs/multisite')  # 使用实际的环境名称
    }
    
    for nodes, cpu_ids, gpu_count in node_info:
        nodes = get_nodes(nodes)
        for node in nodes:
            cpus = count_cpus(cpu_ids)
            config_content[node] = {
                'name': node,
                'cpus': cpus,
                'gpus': gpu_count,
                'memory': '32GB'  # tmp, need calculate
            }
    
    return config_content

def write_config_file(config, file_path):
    with open(file_path, 'w') as config_file:
        config.write(config_file)

def main(slurm_job_id, config_file_path):
    # 读取 Slurm 信息文件
    # with open(slurm_info_path, 'r') as slurm_info_file:
    #     slurm_info = slurm_info_file.read()
    output = subprocess.check_output(['scontrol', 'show', 'jobid', slurm_job_id, '-dd'])
    slurm_info = output.decode().strip()
    print(f"Slurm 信息已读取: {slurm_info}")
    # 解析 Slurm 信息
    config = parse_slurm_info(slurm_info)
    
    # 写入配置文件
    write_config_file(config, config_file_path)
    print(f"配置文件已创建: {config_file_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="解析Slurm信息并创建配置文件。")
    parser.add_argument('--slurm_job_id', type=str, help="Slurm job id")
    parser.add_argument('--slurm_resources_file', type=str, help="配置文件输出路径。")
    args = parser.parse_args()
    
    main(args.slurm_job_id, args.slurm_resources_file)