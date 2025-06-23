import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import subprocess
import logging
import time
import pickle
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def check_file_exists(filepath, step_name, logger):
    if not os.path.exists(filepath):
        logger.error(f"error: required file for {step_name} not found: {filepath}")
        return False
    logger.info(f"found required file: {filepath}")
    return True

def setup_logging(step_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if step_name:
        log_filename = f"{step_name.lower().replace(' ', '_')}_{timestamp}.log"
        logger_name = f"mof_pipeline.{step_name.lower().replace(' ', '_')}"
    else:
        log_filename = f"mof_pipeline_{timestamp}.log"
        logger_name = "mof_pipeline"

    script_dir = os.path.dirname(__file__)
    log_dir = os.path.abspath(os.path.join(script_dir, '..', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    log_filepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized - log file: {log_filepath}")
    return logger


def run_command(cmd, step_name, timeout_hours=24):
    """Runs a command-line tool as a pipeline step."""
    logger = setup_logging(step_name)
    
    logger.info("="*60)
    logger.info(f"Starting: {step_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Timeout: {timeout_hours} hours")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        script_dir = os.path.dirname(__file__)
        log_dir = os.path.abspath(os.path.join(script_dir, '..', 'logs'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subprocess_log = os.path.join(log_dir, f"{step_name.lower().replace(' ', '_')}_output_{timestamp}.log")
        
        logger.info(f"Subprocess output will be saved to: {subprocess_log}")
        
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        env['NUMEXPR_NUM_THREADS'] = '1'
        env['BLAS_NUM_THREADS'] = '1'
        env['LAPACK_NUM_THREADS'] = '1'
        
        with open(subprocess_log, 'w') as log_file:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=log_file,
                stderr=subprocess.STDOUT,  
                text=True,
                timeout=timeout_hours * 3600,
                env=env
            )
        
        elapsed = (time.time() - start_time) / 3600
        logger.info(f"{step_name} completed successfully in {elapsed:.2f} hours")
        logger.info(f"Full output logged to: {subprocess_log}")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - start_time) / 3600
        logger.error(f"{step_name} failed after {elapsed:.2f} hours")
        logger.error(f"Error code: {e.returncode}")
        logger.error(f"Check the subprocess log for detailed error information: {subprocess_log}")
        return False
        
    except subprocess.TimeoutExpired:
        elapsed = (time.time() - start_time) / 3600
        logger.error(f"{step_name} timed out after {elapsed:.2f} hours")
        return False
    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        logger.error(f"{step_name} failed with unexpected error after {elapsed:.2f} hours")
        logger.error(f"Error: {str(e)}")
        return False

def main():    
    # reading data from user
    parser = argparse.ArgumentParser(description='Initializing MOF pipeline')
    parser.add_argument('--mof_data', required=True, help='Path to the MOF dataset csv file')
    parser.add_argument('--output_dir', required=True, help='Path to output directory for adj matrix')
    parser.add_argument('--resolution', default=1.0, type=float, help='Resolution for community detection')
    parser.add_argument('--threshold', default=0.9, type=float, help='Threshold for similarity')
    parser.add_argument('--algorithms', default='louvain', help='Algorithms to run')
    parser.add_argument('--size', required=True, help='size of dataset')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_logger = setup_logging()
    
    mof_network_path = os.path.join(script_dir, 'mof_social_network.py')
    community_analysis_path = os.path.join(script_dir, 'community_network_analysis.py')
    centrality_analysis_path = os.path.join(script_dir, '..', 'scripts', 'analysis_centrality.py')

    adj_file = os.path.join(args.output_dir, f'adjacency_matrix_t{args.threshold}.pkl')
    meta_data_file = os.path.join(args.output_dir, f'adjacency_matrix_metadata_t{args.threshold}.pkl')

    main_logger.info("starting MOF Analysis Pipeline")
    main_logger.info(f"input data: {args.mof_data}")
    main_logger.info(f"output directory: {args.output_dir}")
    main_logger.info(f"parameters: threshold={args.threshold}, resolution={args.resolution}")
    main_logger.info(f"algorithm: {args.algorithms}")

    main_logger.info("="*60)
    main_logger.info("SYSTEM INFORMATION")
    main_logger.info("="*60)
    main_logger.info(f"python version: {sys.version}")
    main_logger.info(f"working directory: {os.getcwd()}")
    main_logger.info(f"script directory: {script_dir}")
    
    if not check_file_exists(args.mof_data, "Pipeline", main_logger):
        main_logger.error("pipeline failed: input data not found")
        return False

    # run the mof_social_network.py script -> adj matrix
    main_logger.info("\nSTEP 1: Generating adjacency matrix...")
    
    step1_cmd = [
        'python', mof_network_path,
        '--input', args.mof_data,
        '--output_dir', args.output_dir,
        '--threshold', str(args.threshold),
        '--resolution', str(args.resolution),
        '--algorithm', args.algorithms,
        '--save_adjacency'  
    ]
    
    main_logger.info(f"Step 1 command: {' '.join(step1_cmd)}")
    
    if not run_command(step1_cmd, "Adjacency Matrix Generation", timeout_hours=48):
        main_logger.error("pipeline failed at Step 1")
        return False
    
    if not check_file_exists(adj_file, "Step 2", main_logger):
        main_logger.error("adjacency matrix not generated properly")
        return False

    # run the community_analysis.py script -> communities
    main_logger.info("\nSTEP 2: Running community analysis...")
    
    base_analysis_dir = os.path.abspath(os.path.join(args.output_dir, '..'))
    community_output_dir = os.path.join(base_analysis_dir, 'comm_analysis')
    os.makedirs(community_output_dir, exist_ok=True)
    
    step2_cmd = [
        'python', community_analysis_path,
        '--adjacency_matrix', adj_file,
        '--metadata', meta_data_file,
        '--data_file', args.mof_data,
        '--output_dir', community_output_dir,
        '--resolution', str(args.resolution)
    ]
    
    main_logger.info(f"Step 2 command: {' '.join(step2_cmd)}")
    
    if not run_command(step2_cmd, "Community Analysis", timeout_hours=12):
        main_logger.error("pipeline failed at Step 2")
        return False
    
    # run the analysis_centrality.py script -> centrality calc
    if os.path.exists(centrality_analysis_path):
        main_logger.info("\nSTEP 3: Running centrality analysis...")
        
        base_dataset_dir = os.path.abspath(os.path.join(args.output_dir, '..', '..'))
        centrality_output_dir = os.path.join(base_dataset_dir, f'integrated_analysis_louvain_t{args.threshold}')
        os.makedirs(centrality_output_dir, exist_ok=True)
        
        community_assignments_file = os.path.join(community_output_dir, 'community_assignments.csv')
        
        step3_cmd = [
            'python', centrality_analysis_path,
            '--adjacency_matrix', adj_file,
            '--communities_file', community_assignments_file,
            '--output_dir', centrality_output_dir
        ]
        
        main_logger.info(f"Step 3 command: {' '.join(step3_cmd)}")
        
        if not run_command(step3_cmd, "Centrality Analysis", timeout_hours=6):
            main_logger.error("Pipeline failed at Step 3")
            return False
    else:
        main_logger.warning("centrality analysis script not found, skipping Step 3")
    
    main_logger.info("\nPIPELINE COMPLETED SUCCESSFULLY!")
    main_logger.info("="*60)
    main_logger.info("RESULTS SUMMARY")
    main_logger.info("="*60)
    main_logger.info(f"Key outputs:")
    main_logger.info(f"   - Adjacency matrix: {adj_file}")
    main_logger.info(f"   - Community analysis: {community_output_dir}")
    if os.path.exists(centrality_analysis_path):
        main_logger.info(f"   - Centrality analysis: {centrality_output_dir}")
    
    log_dir = os.path.abspath(os.path.join(script_dir, '..', 'logs'))
    main_logger.info(f"Log files saved in: {log_dir}")
    
    return True

if __name__ == "__main__":
    main()
