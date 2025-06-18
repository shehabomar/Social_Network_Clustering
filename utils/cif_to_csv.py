import os
import pandas as pd
from pathlib import Path
import numpy as np
import warnings 
from pymatgen.io.cif import CifParser
warnings.filterwarnings('ignore')

def cif_to_csv(cif_dir, output_dir):
    
    cif_files = list(Path(cif_dir).glob('*.cif'))
    print(f"found{len(cif_files)} cif files")
    
    
    data_list = []
    
    for cif_file in cif_files:
        try:
            cif_parser = CifParser(cif_file)
            structure = cif_parser.get_structures()[0]
            
            data = {
                'filename': cif_file.stem,
                'formula': str(structure.formula),
                'reduced_formula': structure.composition.reduced_formula,
                'cell_a': structure.lattice.a,
                'cell_b': structure.lattice.b,
                'cell_c': structure.lattice.c,
                'cell_alpha': structure.lattice.alpha,
                'cell_beta': structure.lattice.beta,
                'cell_gamma': structure.lattice.gamma,
                'cell_volume': structure.lattice.volume,
                'density': structure.density,
                'num_atoms': len(structure.sites),
                'space_group': str(structure.get_space_group_info()[0])
            }
            
            composition = structure.composition
            for element in composition.elements:
                data[f'{element.symbol}_fraction'] = composition.get_atomic_fraction(element)
            
            data_list.append(data)
        except Exception as e:
            print(f'error: {e}')
            
    if data_list:
        df = pd.DataFrame(data_list)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        df.to_csv(output_dir, index=False)
        print(f'saved {len(df)} structures to {output_dir}')
    else:
        print('no valid structures found')    
            
            
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CIF files to CSV format')
    parser.add_argument('input_path', help='Path to CIF file or directory containing CIF files')
    parser.add_argument('--output_dir', default='../data/cif_to_csv', help='Output directory')

    args = parser.parse_args()
    
    cif_to_csv(args.input_path, args.output_dir)

if __name__ == '__main__':
    main()

                